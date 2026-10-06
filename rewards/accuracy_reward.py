"""Accuracy reward: did the agent get the final answer right?

Used by BOTH the GRPO baseline and the cost-aware variant -- the only
difference between those two runs is whether cost_aware_reward is added.

Reward shape (documented in the report):
    +1.0  final answer correct
    +0.2  valid format (an 'Answer: <value>' line exists at all)
     0.0  otherwise

The small format term stabilizes early GRPO training on a 0.6B model
(otherwise almost every rollout scores 0 and advantages are uninformative).
It is tiny compared to the correctness term so it cannot dominate behavior.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewards.parsing import answers_match, extract_answer, extract_completion_text

CORRECT_REWARD = 1.0
FORMAT_REWARD = 0.2


def accuracy_reward(completions, answer, answer_type, alternatives=None, **kwargs):
    """TRL reward function.

    Args (passed by GRPOTrainer from dataset columns):
        completions: list[str] | list[list[dict]]  generated text or conversation per rollout
        answer:      list        gold answers
        answer_type: list[str]   'numeric' | 'string'
        alternatives: list       accepted string variants (may be None entries)
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = extract_completion_text(completion)
        r = 0.0
        pred = extract_answer(text)
        if pred is not None:
            r += FORMAT_REWARD
            alts = []
            if alternatives is not None:
                try:
                    alts = list(alternatives[i] or [])
                except (IndexError, TypeError):
                    alts = []
            if answers_match(pred, answer[i], str(answer_type[i]), alts):
                r += CORRECT_REWARD
        rewards.append(r)
    return rewards


if __name__ == "__main__":
    demo_completions = [
        "Let me compute. 47*83=3901.\nAnswer: 3901",
        "Answer: 3902",
        "I think it is around 3900",
    ]
    demo_answers = [3901, 3901, 3901]
    demo_types = ["numeric", "numeric", "numeric"]
    print(accuracy_reward(demo_completions, demo_answers, demo_types))
    # expected: [1.2, 0.2, 0.0]
