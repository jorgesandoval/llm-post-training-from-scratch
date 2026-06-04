"""Loss functions for every post-training stage, implemented from scratch.

Each function is written to be read: the goal is to expose *exactly* where the
"trick" of each algorithm lives.

Contents:
  * :func:`causal_lm_loss`            -- next-token cross entropy (CPT & SFT).
  * :func:`sequence_logprobs`         -- summed log p(response | prompt).
  * :func:`dpo_loss`                  -- Direct Preference Optimization.
  * :func:`group_normalized_advantage`-- GRPO's value-network-free baseline.
  * :func:`grpo_policy_loss`          -- clipped policy gradient + KL (GRPO).

References:
  * DPO:  Rafailov et al., 2023, "Direct Preference Optimization: Your Language
          Model is Secretly a Reward Model" (https://arxiv.org/abs/2305.18290).
  * GRPO: Shao et al., 2024, "DeepSeekMath" (https://arxiv.org/abs/2402.03300).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token prediction loss (used by CPT and SFT).

    The model predicts token ``t+1`` from tokens ``<= t``, so we shift logits and
    labels by one. Positions whose label is ``IGNORE_INDEX`` (-100) are skipped --
    that is how SFT masks the prompt and trains only on response tokens.

    Args:
        logits: ``(B, T, V)`` raw model outputs.
        labels: ``(B, T)`` target token ids, with -100 where loss is masked.

    Returns:
        Scalar mean cross-entropy over the non-masked positions.
    """
    # Shift so token t predicts token t+1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def sequence_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    """Sum (or average) the log-probabilities of the label tokens per sequence.

    This is the building block for DPO: it answers "under this model, how likely
    is this exact response?". Masked positions (-100) contribute nothing.

    Args:
        logits: ``(B, T, V)`` raw model outputs.
        labels: ``(B, T)`` target token ids, -100 where ignored.
        average: If True, divide by the number of scored tokens (length-normalize).

    Returns:
        ``(B,)`` tensor of (summed or averaged) log-probabilities.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    mask = shift_labels != IGNORE_INDEX
    # Replace -100 with 0 so gather() has a valid index; masked out afterwards.
    safe_labels = shift_labels.clamp_min(0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask  # zero-out masked positions

    summed = token_logp.sum(dim=-1)
    if average:
        counts = mask.sum(dim=-1).clamp_min(1)
        return summed / counts
    return summed


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss (Rafailov et al., 2023, Eq. 7).

    The implicit reward of a response under the policy relative to the reference
    is ``beta * (logp_policy - logp_ref)``. DPO maximizes the margin between the
    chosen and rejected implicit rewards via a logistic (Bradley-Terry) loss::

        L = -log sigmoid( beta * [ (logp_chosen - logp_chosen_ref)
                                 - (logp_rejected - logp_rejected_ref) ] )

    No reward model and no sampling are needed -- this is what makes DPO so much
    simpler than PPO-based RLHF. ``beta`` controls how far the policy may drift
    from the reference (KL regularization strength).

    Args:
        policy_chosen_logps:   ``(B,)`` log p_policy(chosen).
        policy_rejected_logps: ``(B,)`` log p_policy(rejected).
        ref_chosen_logps:      ``(B,)`` log p_ref(chosen).
        ref_rejected_logps:    ``(B,)`` log p_ref(rejected).
        beta: Regularization / temperature.

    Returns:
        Tuple ``(loss, chosen_rewards, rejected_rewards)`` where the rewards are
        the (detached) implicit rewards, handy for logging accuracy/margins.
    """
    # HERE: the per-example log-ratios are the implicit rewards. The reference
    # model anchors the policy so it can't just inflate everything.
    chosen_logratio = policy_chosen_logps - ref_chosen_logps
    rejected_logratio = policy_rejected_logps - ref_rejected_logps

    logits = chosen_logratio - rejected_logratio  # preference margin
    loss = -F.logsigmoid(beta * logits).mean()

    chosen_rewards = (beta * chosen_logratio).detach()
    rejected_rewards = (beta * rejected_logratio).detach()
    return loss, chosen_rewards, rejected_rewards


def group_normalized_advantage(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """GRPO advantage: normalize rewards WITHIN a group of samples for one prompt.

    Given ``G`` sampled responses to the same prompt with rewards ``r_1..r_G``::

        A_i = (r_i - mean(r)) / (std(r) + eps)

    The group mean is the baseline. This is the central GRPO idea (Shao et al.,
    2024): the group itself provides the baseline, so PPO's separate *value
    network* (critic) is eliminated entirely.

    Args:
        rewards: ``(G,)`` rewards for one group, or ``(N, G)`` for a batch of N
            prompts each with G samples.
        eps: Numerical floor for the standard deviation.

    Returns:
        Tensor of the same shape as ``rewards`` containing the advantages.
    """
    if rewards.dim() == 1:
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        return (rewards - mean) / (std + eps)

    # Batched: normalize along the group dimension (last axis).
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_policy_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    token_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_coef: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level clipped policy-gradient objective with a KL penalty (GRPO).

    For each response token we form the probability ratio between the current
    policy and the policy that generated the sample (``old``), then apply PPO's
    clipped surrogate using the group-normalized advantage::

        ratio_t   = exp(logp_t - old_logp_t)
        surrogate = min( ratio_t * A , clip(ratio_t, 1-eps, 1+eps) * A )

    A KL penalty toward the frozen reference keeps the policy from collapsing.
    GRPO uses the unbiased estimator
    ``KL = exp(ref - policy) - (ref - policy) - 1`` (always >= 0).

    Args:
        policy_logps: ``(B, T)`` per-token log-probs under the CURRENT policy.
        old_logps:    ``(B, T)`` per-token log-probs under the sampling policy
            (use ``policy_logps.detach()`` for the single-update / on-policy case).
        ref_logps:    ``(B, T)`` per-token log-probs under the frozen reference.
        advantages:   ``(B,)`` per-sequence advantages (broadcast over tokens).
        token_mask:   ``(B, T)`` 1 for response tokens, 0 for prompt/padding.
        clip_eps: PPO clipping range epsilon.
        kl_coef: Weight on the KL-to-reference penalty.

    Returns:
        Tuple ``(loss, metrics)``.
    """
    adv = advantages.unsqueeze(-1)  # (B, 1) -> broadcast over tokens

    # HERE: the group-relative advantage (no critic) drives the PPO surrogate.
    ratio = torch.exp(policy_logps - old_logps)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)

    # Unbiased KL(policy || ref) estimator, per token (Schulman's k3 estimator).
    log_ratio_ref = ref_logps - policy_logps
    per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0

    per_token_loss = -(surrogate - kl_coef * per_token_kl)

    # Mask to response tokens and average over the real (non-padded) tokens.
    masked = per_token_loss * token_mask
    denom = token_mask.sum().clamp_min(1.0)
    loss = masked.sum() / denom

    metrics = {
        "kl": (per_token_kl * token_mask).sum().item() / denom.item(),
        "ratio_mean": (ratio * token_mask).sum().item() / denom.item(),
        "adv_mean": advantages.mean().item(),
    }
    return loss, metrics


def per_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of the realized next token (no masking, no reduction).

    Returns a ``(B, T-1)`` tensor aligned with ``input_ids[:, 1:]``. Useful for
    GRPO, where we need token-level log-probs before applying the response mask.
    """
    shift_logits = logits[:, :-1, :]
    shift_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    return log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
