import random
import json
from config import VOCAB, INV_VOCAB, PAD_TOKEN, START_TOKEN, END_TOKEN, GRADES, hold_to_coords
from dataset import MoonBoardDataset

"""
Modifier for dataset to introduce "wrong" problems into the dataset as a <difficulty> token.
This patch generates a 5% "fake" problems into the dataset with an initial difficulty token
labeled "<WRONG>". The generated problems are real correct problems with an added 2-4 holds
appended to the sequence AFTER the END_TOKEN — making them structurally malformed — to help 
guide the model towards learning that the <WRONG> tag is associated with bad/over-extended problems.
"""

WRONG_TOKEN_STR = '<WRONG>'
ADDED_HOLD_RANGE = (2, 4)


def generate_wrong_problems(dataset: MoonBoardDataset, fraction: float = 0.05) -> list:
    """
    Takes an existing MoonBoardDataset and generates a set of "wrong" problems.
    Each wrong problem is a copy of a real problem with:
      - Its grade token replaced by '<WRONG>'
      - 2–4 random extra holds appended after the last valid hold (simulating a physically bad problem)

    Args:
        dataset:  A loaded MoonBoardDataset instance.
        fraction: The fraction of the dataset size to generate as wrong problems (default 5%).

    Returns:
        A list of dicts with the same schema as dataset.problems:
        [{'holds': [...], 'grade': '<WRONG>'}, ...]
    """
    num_to_generate = max(1, int(len(dataset.problems) * fraction))

    # All valid hold names from the vocabulary (exclude special tokens and grade tokens)
    all_holds = [
        token_str for token_str, token_id in VOCAB.items()
        if token_str not in ['<PAD>', '<START>', '<END>'] and token_str not in GRADES
    ]

    wrong_problems = []
    source_problems = dataset.problems  # list of {'holds': [...], 'grade': '<Xgrade>'}

    for _ in range(num_to_generate):
        # Pick a random real problem to corrupt
        base = random.choice(source_problems)
        holds = list(base['holds'])  # copy so we don't mutate original

        # Append 2–4 random extra holds that aren't already in the sequence
        # to make the problem physically "wrong" (too many moves / illogical additions)
        available = [h for h in all_holds if h not in holds]
        num_extra = random.randint(*ADDED_HOLD_RANGE)
        extra_holds = random.sample(available, min(num_extra, len(available)))
        holds.extend(extra_holds)

        wrong_problems.append({
            'holds': holds,
            'grade': WRONG_TOKEN_STR,
        })

    return wrong_problems


def inject_wrong_problems(dataset: MoonBoardDataset, fraction: float = 0.05) -> MoonBoardDataset:
    """
    Injects wrong problems directly into a dataset's .problems list in-place.
    Call this before wrapping the dataset in a DataLoader.

    Also registers the <WRONG> token into VOCAB / INV_VOCAB if not already present.

    Args:
        dataset:  A loaded MoonBoardDataset instance.
        fraction: Fraction of dataset size to inject as wrong problems.

    Returns:
        The same dataset object with wrong problems appended.
    """
    # Register <WRONG> token in the vocabulary if missing
    if WRONG_TOKEN_STR not in VOCAB:
        new_id = max(VOCAB.values()) + 1
        VOCAB[WRONG_TOKEN_STR] = new_id
        INV_VOCAB[new_id] = WRONG_TOKEN_STR
        print(f"Registered '{WRONG_TOKEN_STR}' as token ID {new_id}")

    wrong_problems = generate_wrong_problems(dataset, fraction=fraction)
    dataset.problems.extend(wrong_problems)

    print(f"Injected {len(wrong_problems)} wrong problems into dataset "
          f"(total size now: {len(dataset.problems)}).")
    return dataset


if __name__ == '__main__':
    # Quick smoke test
    ds = MoonBoardDataset('problems MoonBoard Masters 2019 40.json')
    print(f"Original dataset size: {len(ds)}")

    ds = inject_wrong_problems(ds, fraction=0.05)
    print(f"Dataset size after injection: {len(ds)}")

    # Show a couple of the generated wrong problems
    wrong_samples = [p for p in ds.problems if p['grade'] == WRONG_TOKEN_STR]
    for i, sample in enumerate(wrong_samples[:3]):
        print(f"\n[Wrong Problem {i+1}] grade={sample['grade']}, holds={sample['holds']}")