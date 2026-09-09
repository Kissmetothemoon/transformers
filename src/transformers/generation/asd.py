# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Approximate Speculative Decoding (ASD) acceptance policy for greedy assisted decoding.

Reference: https://arxiv.org/abs/2608.03447
"""

import torch


def _asd_greedy_acceptance(
    candidate_new_tokens: torch.Tensor,
    selected_tokens: torch.Tensor,
    new_logits: torch.Tensor,
    candidate_length: int,
    regret_spent: float,
    asd_budget: float,
    asd_local_ratio: float,
    asd_max_mismatches: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Approximate Speculative Decoding (ASD) acceptance for greedy assisted decoding
    (https://arxiv.org/abs/2608.03447). A draft token `x_i` is accepted while ALL of the following hold:

    1. request-level cumulative regret stays within `asd_budget` (regret `r_i = max_v z_i(v) - z_i(x_i)`
       against the target logits `z_i`; `r_i = 0` iff `x_i` is the target argmax);
    2. the local regret satisfies `r_i / (K - i) <= asd_local_ratio` (suffix-value weighting: later draft
       positions get less slack);
    3. the number of relaxed (non-argmax) tokens in the block does not exceed `asd_max_mismatches`.

    Acceptance stops at the first infeasible position. With `asd_budget = 0` or `asd_max_mismatches = 0`
    this reduces exactly to strict greedy verification.

    Returns the number of accepted draft tokens, a boolean mask of the relaxed (draft-committed) positions,
    and the regret spent in this block.
    """
    draft_logits = new_logits[:, :candidate_length, :].gather(-1, candidate_new_tokens.unsqueeze(-1)).squeeze(-1)
    regrets = new_logits[:, :candidate_length, :].max(dim=-1).values - draft_logits
    mismatches = regrets > 0

    suffix_values = torch.arange(candidate_length, 0, -1, dtype=regrets.dtype, device=regrets.device)
    feasible = (
        (regret_spent + regrets.cumsum(dim=-1) <= asd_budget)
        & (regrets / suffix_values <= asd_local_ratio)
        & (mismatches.cumsum(dim=-1) <= asd_max_mismatches)
    )

    accepted_mask = (~feasible).cumsum(dim=-1) < 1
    n_matches = accepted_mask.sum()
    relaxed_mask = mismatches & accepted_mask
    step_regret = (regrets * accepted_mask).sum().item()
    return n_matches, relaxed_mask, step_regret
