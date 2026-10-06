"""Cost-aware / capability-aware reward for TinyToolRL.

Implements the three reward modes defined in the project research design:

    1. Outcome (Baseline 1):
       R_outcome = R_correct
       (no penalty on tool calls; purely outcome-driven)

    2. FixedCost / OTC-style (Baseline 2, arXiv:2504.14870):
       R_fixed = R_correct - lambda * I_tool
       (uniform penalty lambda whenever the calculator is invoked)

    3. CapabilityAware (Ours):
       R_cap = R_correct - lambda * c(x) * I_tool
       where c(x) in [0, 1] is the frozen SFT model pi_0 accuracy without tools
       (estimated via sampling K=3 times, todo item E).
       - When c(x) = 1.0 (model capable of solving directly): full penalty lambda.
       - When c(x) = 0.0 (model incapable without tools): ZERO penalty (legitimate tool use).
       - When c(x) in (0, 1): penalty proportional to baseline capability.

I_tool is binary indicator (1.0 if agent issued any tool call, 0.0 otherwise).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.accuracy_reward import accuracy_reward
from rewards.parsing import check_has_tool_call, extract_completion_text, has_tool_call

VALID_MODES = ("outcome", "fixed", "cap", "capability")


def compute_tool_cost(
    completion,
    mode: str,
    capability: float | None = None,
    needs_tool: bool | None = None,
) -> float:
    """Compute the unscaled tool cost C per rollout.

    Total rollout reward: R = R_correct - lambda * C.

    Returns:
        float: cost C >= 0.
          - mode == "outcome": always 0.0
          - mode == "fixed": 1.0 if tool called, else 0.0
          - mode in ("cap", "capability"): c(x) if tool called, else 0.0
    """
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown reward mode {mode!r}. Valid modes: {VALID_MODES}")

    if mode == "outcome":
        return 0.0

    called = check_has_tool_call(completion)
    if not called:
        return 0.0

    if mode == "fixed":
        return 1.0

    # mode in ("cap", "capability")
    if capability is not None:
        try:
            return float(capability)
        except (ValueError, TypeError):
            pass

    # Fallback if capability column not yet populated in dataset (pre-Task E):
    # approximate c(x) from binary needs_tool (needs_tool=True -> c=0.0; needs_tool=False -> c=1.0)
    if needs_tool is not None:
        return 0.0 if bool(needs_tool) else 1.0

    # Default if neither is provided: full cost
    return 1.0


def make_tool_penalty_reward(mode: str, penalty_lambda: float):
    """Factory returning a standalone TRL penalty reward function.

    Returns negative penalty values: -lambda * C <= 0.
    """
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown reward mode {mode!r}. Valid modes: {VALID_MODES}")

    def tool_penalty_reward(
        completions,
        capability=None,
        needs_tool=None,
        **kwargs,
    ):
        rewards = []
        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)
            cap = None
            if capability is not None:
                try:
                    cap = capability[i]
                except (IndexError, TypeError):
                    cap = None

            nt = None
            if needs_tool is not None:
                try:
                    nt = needs_tool[i]
                except (IndexError, TypeError):
                    nt = None

            cost = compute_tool_cost(
                completion=completion,
                mode=mode,
                capability=cap,
                needs_tool=nt,
            )
            rewards.append(-penalty_lambda * cost)
        return rewards

    tool_penalty_reward.__name__ = f"penalty_{mode}_lam{penalty_lambda:g}"
    tool_penalty_reward.mode = mode
    tool_penalty_reward.penalty_lambda = penalty_lambda
    return tool_penalty_reward


def get_reward_functions(mode: str, penalty_lambda: float = 0.1) -> list:
    """Return the list of TRL reward functions for GRPOTrainer.

    For 'outcome' (or lambda=0), returns [accuracy_reward].
    For 'fixed' or 'cap', returns [accuracy_reward, tool_penalty_reward].
    Using separate reward functions allows TRL to log accuracy and penalty
    metrics independently to WandB / TensorBoard.
    """
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown reward mode {mode!r}. Valid modes: {VALID_MODES}")

    if mode == "outcome" or penalty_lambda <= 0.0:
        return [accuracy_reward]

    penalty_fn = make_tool_penalty_reward(mode=mode, penalty_lambda=penalty_lambda)
    return [accuracy_reward, penalty_fn]


if __name__ == "__main__":
    demo_completions = [
        '<tool_call>{"name": "calculator", "arguments": {"expression": "7+5"}}</tool_call> Answer: 12',
        "7 + 5 = 12.\nAnswer: 12",
        '<tool_call>{"name": "calculator", "arguments": {"expression": "47*83"}}</tool_call> Answer: 3901',
        "Answer: 3901",
    ]
    # Capabilities: [1.0 (easy), 1.0 (easy), 0.0 (hard), 0.0 (hard)]
    caps = [1.0, 1.0, 0.0, 0.0]

    lam = 0.1

    fn_outcome = make_tool_penalty_reward("outcome", lam)
    fn_fixed = make_tool_penalty_reward("fixed", lam)
    fn_cap = make_tool_penalty_reward("cap", lam)

    out_outcome = fn_outcome(demo_completions, capability=caps)
    out_fixed = fn_fixed(demo_completions, capability=caps)
    out_cap = fn_cap(demo_completions, capability=caps)

    print("Outcome penalties:", out_outcome)
    assert out_outcome == [0.0, 0.0, 0.0, 0.0], f"unexpected outcome: {out_outcome}"

    print("Fixed penalties:  ", out_fixed)
    assert out_fixed == [-0.1, 0.0, -0.1, 0.0], f"unexpected fixed: {out_fixed}"

    print("Cap penalties:    ", out_cap)
    assert out_cap == [-0.1, 0.0, 0.0, 0.0], f"unexpected cap: {out_cap}"

    print("\nAll cost-aware reward tests PASSED!")
