"""Dataset builders for every post-training stage.

Each stage consumes data in a different shape, and the *shape of the labels* is
where much of the conceptual difference lives:

  * CPT : raw text -> fixed-length blocks, loss on ALL tokens.
  * SFT : (prompt, response) -> labels masked (-100) on the prompt.
  * DPO : (prompt, chosen, rejected) -> two masked sequences per example.
  * GRPO: procedurally generated arithmetic, with a verifiable reward function.

We keep one simple, explicit prompt format shared by SFT/DPO/GRPO so the chained
pipeline stays consistent. The format is deliberately hand-rolled (rather than
relying on a tokenizer chat template) to keep everything visible and didactic.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100

# A minimal, explicit instruction format reused across SFT / DPO / GRPO.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Render an instruction into the shared prompt prefix (no response yet)."""
    return PROMPT_TEMPLATE.format(instruction=instruction)


# --------------------------------------------------------------------------- #
# CPT: Continued Pre-Training
# --------------------------------------------------------------------------- #
class CPTDataset(Dataset):
    """Pack raw documents into fixed-length blocks for next-token prediction.

    There is no prompt/response structure here: every token is a training target
    (labels == input_ids), exactly like the original pre-training objective.
    """

    def __init__(self, texts: list[str], tokenizer, block_size: int = 256):
        self.block_size = block_size
        eos = tokenizer.eos_token or ""
        # Concatenate all documents, separated by EOS, then chunk into blocks.
        joined = eos.join(texts) + eos
        ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
        self.blocks: list[list[int]] = [
            ids[i : i + block_size]
            for i in range(0, len(ids) - block_size + 1, block_size)
        ]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        # CPT trains on every token => labels are just the inputs.
        return {"input_ids": ids, "labels": ids.clone()}


# --------------------------------------------------------------------------- #
# SFT: Supervised Fine-Tuning
# --------------------------------------------------------------------------- #
class SFTDataset(Dataset):
    """(prompt, response) pairs with the prompt tokens masked out of the loss.

    Each example yields ``input_ids`` (prompt + response + EOS) and ``labels``
    where every PROMPT position is set to -100, so the loss is computed only on
    the response. This is the defining difference between SFT and CPT.
    """

    def __init__(self, pairs: list[dict[str, str]], tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = pairs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = format_prompt(ex["instruction"])
        response = ex["response"] + (self.tokenizer.eos_token or "")

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + response_ids)[: self.max_len]
        # HERE: mask the prompt with -100 so only response tokens contribute.
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[: self.max_len]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# DPO: Direct Preference Optimization
# --------------------------------------------------------------------------- #
class DPODataset(Dataset):
    """(prompt, chosen, rejected) triples.

    Returns prompt-masked sequences for BOTH the chosen and rejected responses so
    that :func:`src.losses.sequence_logprobs` scores only the response tokens.
    """

    def __init__(self, triples: list[dict[str, str]], tokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = triples

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, prompt_ids: list[int], response: str):
        response_ids = self.tokenizer(
            response + (self.tokenizer.eos_token or ""), add_special_tokens=False
        )["input_ids"]
        input_ids = (prompt_ids + response_ids)[: self.max_len]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[: self.max_len]
        return input_ids, labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer(
            format_prompt(ex["prompt"]), add_special_tokens=False
        )["input_ids"]

        chosen_ids, chosen_labels = self._encode(prompt_ids, ex["chosen"])
        rejected_ids, rejected_labels = self._encode(prompt_ids, ex["rejected"])

        return {
            "chosen_input_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Collation: right-pad a batch of variable-length sequences
# --------------------------------------------------------------------------- #
def pad_collate(batch: list[dict[str, torch.Tensor]], pad_token_id: int):
    """Collate a batch of ``{input_ids, labels}`` dicts with right padding.

    ``input_ids`` are padded with ``pad_token_id`` and ``labels`` with -100 so
    that padding never contributes to the loss. Also returns ``attention_mask``.
    """
    keys = batch[0].keys()
    max_len = max(item["input_ids"].size(0) for item in batch)
    out: dict[str, torch.Tensor] = {}

    for key in keys:
        pad_value = IGNORE_INDEX if "labels" in key else pad_token_id
        rows = []
        for item in batch:
            seq = item[key]
            pad_len = max_len - seq.size(0)
            rows.append(torch.cat([seq, torch.full((pad_len,), pad_value, dtype=seq.dtype)]))
        out[key] = torch.stack(rows)

    out["attention_mask"] = (out["input_ids"] != pad_token_id).long()
    return out


# --------------------------------------------------------------------------- #
# GRPO: procedurally generated arithmetic with a verifiable reward
# --------------------------------------------------------------------------- #
GRPO_SYSTEM_HINT = (
    "Solve the problem. Show your reasoning inside <reasoning></reasoning> "
    "and give the final number inside <answer></answer>."
)

# A single worked example ("one-shot") that primes the answer FORMAT. A weak base
# model almost never emits the tags zero-shot, which would leave GRPO with no
# reward signal; one demonstration is enough to make the format appear often
# enough (with sampling variance) for the group-relative advantage to act on.
# (Empirically, one-shot >> zero-shot or two-shot for this 135M model.)
ARITHMETIC_FEWSHOT = (
    "What is 2 + 3?\n<reasoning>2 plus 3 is 5.</reasoning><answer>5</answer>\n\n"
)

_ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


@dataclass
class ArithmeticProblem:
    """A single verifiable arithmetic task."""

    question: str
    answer: int

    def prompt(self, fewshot: bool = True) -> str:
        shot = ARITHMETIC_FEWSHOT if fewshot else ""
        instruction = f"{GRPO_SYSTEM_HINT}\n\n{shot}What is {self.question}?"
        return format_prompt(instruction)


def generate_arithmetic_problems(
    n: int,
    max_operand: int = 20,
    ops: tuple[str, ...] = ("+", "-"),
    seed: int | None = None,
) -> list[ArithmeticProblem]:
    """Procedurally generate ``n`` arithmetic problems with known answers."""
    rng = random.Random(seed)
    problems: list[ArithmeticProblem] = []
    for _ in range(n):
        a = rng.randint(0, max_operand)
        b = rng.randint(0, max_operand)
        op = rng.choice(ops)
        answer = a + b if op == "+" else a - b
        problems.append(ArithmeticProblem(question=f"{a} {op} {b}", answer=answer))
    return problems


def extract_answer(text: str) -> int | None:
    """Pull the integer inside the last ``<answer>...</answer>`` tag, if any."""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def arithmetic_reward(generation: str, target: int) -> float:
    """Verifiable, *graded* reward for the GRPO arithmetic task.

    The reward is deterministic and checkable (no learned reward model), and it is
    shaped into several components so a weak model gets a usable gradient toward
    the desired behavior instead of an all-or-nothing signal:

    * +0.1  for well-formed ``<reasoning></reasoning>`` tags,
    * +0.2  for well-formed ``<answer></answer>`` tags,
    * +0.1  for putting *some* parseable integer in the answer tags,
    * +1.0  for the answer being correct.

    Maximum reward is therefore 1.4 (perfect), with partial credit for adopting
    the format even when the arithmetic is wrong.
    """
    reward = 0.0
    if "<reasoning>" in generation and "</reasoning>" in generation:
        reward += 0.1
    if "<answer>" in generation and "</answer>" in generation:
        reward += 0.2

    predicted = extract_answer(generation)
    if predicted is not None:
        reward += 0.1                       # produced a parseable answer
        if predicted == target:
            reward += 1.0                   # HERE: the verifiable correctness term
    return reward
