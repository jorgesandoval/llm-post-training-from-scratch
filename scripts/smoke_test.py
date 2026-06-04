"""Offline smoke test for the from-scratch algorithms (no model download).

Builds a tiny Llama model from config and a fake tokenizer so we can exercise
LoRA injection, the loss functions, the dataset builders, and one step of each
training loop without touching the network. Run with the project venv:

    source .venv/bin/activate && python scripts/smoke_test.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.lora import (inject_lora, lora_parameters, count_parameters,
                      LoRALinear, lora_state_dict, load_lora_state_dict)
from src.losses import (causal_lm_loss, sequence_logprobs, dpo_loss,
                        group_normalized_advantage, grpo_policy_loss,
                        per_token_logprobs)
from src.data import (CPTDataset, SFTDataset, DPODataset, pad_collate,
                      generate_arithmetic_problems, arithmetic_reward)

torch.manual_seed(0)
ok = lambda name: print(f"  [ok] {name}")


# --------------------------------------------------------------------------- #
# A minimal fake tokenizer with just the interface our code uses.
# --------------------------------------------------------------------------- #
class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 1
    pad_token = "<eos>"
    pad_token_id = 1

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        # Encode each character to a small vocab id (deterministic, toy).
        ids = [(ord(c) % 50) + 2 for c in text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [f"t{int(i)}" for i in ids]


def build_tiny_model():
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128)
    return LlamaForCausalLM(cfg)


def test_lora():
    model = build_tiny_model()
    ids = torch.randint(2, 60, (2, 16))
    with torch.no_grad():
        base_logits = model(ids).logits

    n = inject_lora(model, target_modules=("q_proj", "v_proj"), r=4, alpha=8)
    assert n == 2 * model.config.num_hidden_layers, n
    trainable, total = count_parameters(model)
    assert 0 < trainable < total
    # Identity at init (B = 0).
    with torch.no_grad():
        lora_logits = model(ids).logits
    assert torch.allclose(base_logits, lora_logits, atol=1e-5), "LoRA not identity at init"
    # Only LoRA params are trainable.
    n_trainable_tensors = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_trainable_tensors == 2 * n, n_trainable_tensors
    # Save/load round-trip.
    sd = lora_state_dict(model)
    for v in sd.values():
        v.add_(0.123)
    load_lora_state_dict(model, sd)
    ok("lora: inject, identity-at-init, param counts, state-dict round-trip")
    return model


def test_losses():
    B, T, V = 2, 10, 64
    logits = torch.randn(B, T, V, requires_grad=True)
    labels = torch.randint(0, V, (B, T))
    labels[:, :3] = -100  # mask a prefix
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    assert loss.item() > 0 and logits.grad is not None
    ok("causal_lm_loss: positive scalar, backprops")

    lp = sequence_logprobs(logits.detach(), labels)
    assert lp.shape == (B,) and (lp <= 0).all()
    ok("sequence_logprobs: shape (B,), all <= 0")

    # DPO: identical policy/ref -> loss == -log(0.5), zero margin.
    z = torch.zeros(B)
    l, cr, rr = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(l.item() - 0.6931) < 1e-3, l.item()
    # Chosen strongly preferred -> lower loss.
    l2, _, _ = dpo_loss(torch.ones(B), -torch.ones(B), z, z, beta=1.0)
    assert l2.item() < l.item()
    ok("dpo_loss: -log(0.5) at parity, decreases when chosen preferred")

    # Group advantage: zero mean within group, std-normalized.
    r = torch.tensor([0.0, 1.0, 2.0, 3.0])
    a = group_normalized_advantage(r)
    assert abs(a.mean().item()) < 1e-5
    batched = group_normalized_advantage(r.view(2, 2))
    assert batched.shape == (2, 2)
    ok("group_normalized_advantage: zero-mean per group, batched shape")

    # GRPO policy loss shapes.
    pol = torch.randn(B, T - 1, requires_grad=True)
    old = pol.detach().clone()
    ref = torch.randn(B, T - 1)
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(B, T - 1)
    gl, metrics = grpo_policy_loss(pol, old, ref, adv, mask)
    gl.backward()
    assert pol.grad is not None and "kl" in metrics
    ok("grpo_policy_loss: backprops, returns metrics")

    ptl = per_token_logprobs(logits.detach(), labels.clamp_min(0))
    assert ptl.shape == (B, T - 1)
    ok("per_token_logprobs: shape (B, T-1)")


def test_datasets():
    tok = FakeTokenizer()
    cpt = CPTDataset(["hello world", "foo bar baz", "lorem ipsum dolor"], tok, block_size=8)
    item = cpt[0]
    assert torch.equal(item["input_ids"], item["labels"])  # CPT: all tokens are labels
    ok("CPTDataset: labels == input_ids, fixed-length blocks")

    sft = SFTDataset([{"instruction": "hi", "response": "hello there"}], tok)
    s = sft[0]
    assert (s["labels"] == -100).any() and (s["labels"] != -100).any()
    ok("SFTDataset: prompt masked with -100, response kept")

    dpo = DPODataset([{"prompt": "q", "chosen": "good", "rejected": "bad"}], tok)
    d = dpo[0]
    assert {"chosen_input_ids", "rejected_input_ids"} <= set(d)
    batch = pad_collate([{"input_ids": s["input_ids"], "labels": s["labels"]},
                         {"input_ids": d["chosen_input_ids"], "labels": d["chosen_labels"]}],
                        pad_token_id=tok.pad_token_id)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    ok("DPODataset + pad_collate: padded batch shapes line up")

    probs = generate_arithmetic_problems(5, max_operand=9, seed=1)
    full = arithmetic_reward(
        f"<reasoning>x</reasoning><answer>{probs[0].answer}</answer>", probs[0].answer)
    assert len(probs) == 5 and abs(full - 1.4) < 1e-6, full
    ok("arithmetic problems + graded reward")


def test_training_steps(model):
    """One optimizer step for each stage's objective on the tiny model."""
    opt = torch.optim.AdamW(lora_parameters(model), lr=1e-3)

    # CPT/SFT step.
    ids = torch.randint(2, 60, (2, 12))
    labels = ids.clone(); labels[:, :4] = -100
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    loss = causal_lm_loss(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    ok(f"CPT/SFT step: loss={loss.item():.3f}")

    # DPO step (model as policy, frozen clone as reference).
    ref = build_tiny_model()
    inject_lora(ref, target_modules=("q_proj", "v_proj"), r=4, alpha=8)
    for p in ref.parameters():
        p.requires_grad_(False)
    def sl(m, x):
        lg = m(input_ids=x, attention_mask=torch.ones_like(x)).logits
        lab = x.clone(); lab[:, :4] = -100
        return sequence_logprobs(lg, lab)
    l, _, _ = dpo_loss(sl(model, ids), sl(model, ids[:, :10]),
                       sl(ref, ids).detach(), sl(ref, ids[:, :10]).detach(), beta=0.1)
    opt.zero_grad(); l.backward(); opt.step()
    ok(f"DPO step: loss={l.item():.3f}")

    # GRPO step.
    seqs = torch.randint(2, 60, (4, 14))
    attn = torch.ones_like(seqs)
    mask = torch.zeros(4, 14); mask[:, 6:] = 1.0
    rewards = torch.tensor([0.0, 1.0, 0.2, 1.2])
    adv = group_normalized_advantage(rewards.view(2, 2)).view(-1)
    cur = per_token_logprobs(model(input_ids=seqs, attention_mask=attn).logits, seqs)
    old = cur.detach()
    refl = per_token_logprobs(ref(input_ids=seqs, attention_mask=attn).logits, seqs).detach()
    gl, m = grpo_policy_loss(cur, old, refl, adv, mask[:, 1:])
    opt.zero_grad(); gl.backward(); opt.step()
    ok(f"GRPO step: loss={gl.item():.3f}, kl={m['kl']:.4f}")


if __name__ == "__main__":
    print("LoRA");      model = test_lora()
    print("Losses");    test_losses()
    print("Datasets");  test_datasets()
    print("Training");  test_training_steps(model)
    print("\nALL SMOKE TESTS PASSED")
