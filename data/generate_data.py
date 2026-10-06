"""Synthetic data generator for TinyToolRL.

Creates four splits under data/:

    train.jsonl        -- 600 items, 50% need the tool   (SFT source + GRPO prompts)
    eval.jsonl         -- 150 items, balanced             (main in-distribution eval)
    eval_ood.jsonl     -- 120 items, BIGGER numbers       (difficulty generalization)
    eval_ratio20.jsonl -- 120 items, only 20% need tool   (tool-need ratio shift)

Three categories:

    hard_arith  (needs_tool=True)   multi-digit / multi-step arithmetic where a
                                    0.6B model should NOT be trusted to compute
                                    in its head. Calling the calculator is free.
    easy_arith  (needs_tool=False)  single-digit and pattern arithmetic the
                                    model can do directly. Calling costs lambda.
    common_sense(needs_tool=False)  factual / counting questions where the
                                    calculator is clearly irrelevant.

Every item records:
    question, answer, answer_type ('numeric'|'string'), alternatives,
    category, needs_tool, expression (gold calculator input, hard_arith only)

Usage:
    python data/generate_data.py --out_dir data --seed 42
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import save_jsonl
from tools.calculator import format_result, safe_eval

# --------------------------------------------------------------------------
# hard arithmetic (needs_tool = True)
# --------------------------------------------------------------------------

_EXPR_PHRASINGS = [
    "Calculate {e}.",
    "What is {e}?",
    "Compute the value of {e}.",
    "How much is {e}?",
    "Find the result of {e}.",
]


def _display(expr: str) -> str:
    """Python expression -> human display form ('47 * 83' -> '47 × 83')."""
    return expr.replace("*", " × ").replace("/", " ÷ ").replace(" - ", " − ")


def _phrasings_for(rng, expr: str) -> str:
    return rng.choice(_EXPR_PHRASINGS).format(e=_display(expr))


def _hard_item(rng, big: bool):
    """One hard-arithmetic item. big=True -> OOD magnitude (3-4 digits)."""
    kind = rng.choice(
        ["mul", "sum_mul", "two_products", "square_add", "percent",
         "div_exact", "word_shelves", "word_speed", "word_tickets",
         "word_factory", "handshakes"]
    )
    lo2, hi2 = (100, 999) if big else (23, 99)
    if kind == "mul":
        a, b = rng.randint(lo2, hi2), rng.randint(23, 99)
        expr = f"{a} * {b}"
        q = _phrasings_for(rng, expr)
    elif kind == "sum_mul":
        a = rng.randint(100, 899) if big else rng.randint(34, 199)
        b = rng.randint(100, 899) if big else rng.randint(25, 149)
        c = rng.randint(6, 29) if big else rng.randint(3, 19)
        expr = f"({a} + {b}) * {c}"
        q = _phrasings_for(rng, expr)
    elif kind == "two_products":
        a, b = rng.randint(lo2, hi2), rng.randint(13, 59)
        c, d = rng.randint(13, 99), rng.randint(13, 59)
        expr = f"{a} * {b} + {c} * {d}"
        q = _phrasings_for(rng, expr)
    elif kind == "square_add":
        a = rng.randint(30, 120) if big else rng.randint(20, 59)
        b = rng.randint(101, 999) if big else rng.randint(10, 299)
        expr = f"{a} ** 2 + {b}"
        q = _phrasings_for(rng, f"{a}^2 + {b}")
    elif kind == "percent":
        n = rng.randrange(200, 2400, 100) if big else rng.randrange(100, 1200, 100)
        p = rng.randrange(5, 100, 5)
        expr = f"{n} * {p} / 100"
        q = f"What is {p}% of {n}?"
    elif kind == "div_exact":
        b = rng.randint(13, 49) if big else rng.randint(6, 29)
        quot = rng.randint(35, 199) if big else rng.randint(13, 99)
        a = b * quot
        expr = f"{a} / {b}"
        q = _phrasings_for(rng, expr)
    elif kind == "word_shelves":
        n = rng.randint(25, 89) if big else rng.randint(12, 49)
        m = rng.randint(21, 59) if big else rng.randint(13, 39)
        expr = f"{n} * {m}"
        q = (f"A library has {n} shelves, and each shelf holds {m} books. "
             f"How many books are there in total?")
    elif kind == "word_speed":
        v = rng.randint(45, 129) if big else rng.randint(35, 95)
        h = rng.randint(6, 19) if big else rng.randint(3, 12)
        d = v * h
        expr = f"{d} / {h}"
        q = (f"A car travels {d} kilometers in {h} hours at a constant speed. "
             f"What is its speed in kilometers per hour?")
    elif kind == "word_tickets":
        p = rng.randint(23, 79) if big else rng.randint(13, 49)
        n = rng.randint(21, 69) if big else rng.randint(12, 39)
        expr = f"{p} * {n}"
        q = (f"Each concert ticket costs {p} dollars. "
             f"How much do {n} tickets cost, in dollars?")
    elif kind == "word_factory":
        r = rng.randint(305, 1995) if big else rng.randint(105, 495)
        d = rng.randint(15, 59) if big else rng.randint(6, 29)
        expr = f"{r} * {d}"
        q = (f"A factory produces {r} widgets per day. "
             f"How many widgets does it produce in {d} days?")
    else:  # handshakes
        n = rng.randint(20, 45) if big else rng.randint(10, 29)
        expr = f"{n} * ({n} - 1) / 2"
        q = (f"In a group of {n} people, every person shakes hands exactly "
             f"once with every other person. How many handshakes happen in total?")

    ans = format_result(safe_eval(expr))
    return {
        "category": "hard_arith", "needs_tool": True, "question": q,
        "expression": expr, "answer": ans, "answer_type": "numeric",
        "alternatives": [],
    }


# --------------------------------------------------------------------------
# easy arithmetic (needs_tool = False)
# --------------------------------------------------------------------------

def _easy_item(rng):
    kind = rng.choice(["add", "sub", "times_table", "round_add", "double", "half"])
    if kind == "add":
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        q, ans = f"What is {a} + {b}?", a + b
    elif kind == "sub":
        a = rng.randint(6, 30)
        b = rng.randint(1, a - 1)
        q, ans = f"What is {a} - {b}?", a - b
    elif kind == "times_table":
        a, b = rng.randint(2, 9), rng.randint(2, 9)
        q = rng.choice([f"What is {a} × {b}?", f"Calculate {a} times {b}."])
        ans = a * b
    elif kind == "round_add":
        a = rng.randrange(50, 500, 50)
        b = rng.randrange(50, 500, 50)
        q, ans = f"What is {a} + {b}?", a + b
    elif kind == "double":
        a = rng.randint(3, 25)
        q, ans = f"What is double {a}?", 2 * a
    else:
        a = rng.randint(2, 15)
        q, ans = f"What is half of {2 * a}?", a
    return {
        "category": "easy_arith", "needs_tool": False, "question": q,
        "expression": None, "answer": str(ans), "answer_type": "numeric",
        "alternatives": [],
    }


# --------------------------------------------------------------------------
# common sense (needs_tool = False)
# --------------------------------------------------------------------------

# (question, answer, answer_type, alternatives)
_COMMON_HANDWRITTEN = [
    ("What is the capital city of France?", "Paris", "string", []),
    ("What is the capital city of Japan?", "Tokyo", "string", []),
    ("What is the capital city of Italy?", "Rome", "string", []),
    ("What is the capital city of Egypt?", "Cairo", "string", []),
    ("What is the capital city of Germany?", "Berlin", "string", []),
    ("What is the capital city of Spain?", "Madrid", "string", []),
    ("What is the capital city of Canada?", "Ottawa", "string", []),
    ("What is the capital city of China?", "Beijing", "string", []),
    ("What is the capital city of Australia?", "Canberra", "string", []),
    ("What color is the sky on a clear day?", "blue", "string", []),
    ("What color is a ripe banana?", "yellow", "string", []),
    ("What color is fresh grass?", "green", "string", []),
    ("What color is snow?", "white", "string", []),
    ("Which is the largest planet in our solar system?", "Jupiter", "string", []),
    ("Which planet is closest to the Sun?", "Mercury", "string", []),
    ("Which planet do we live on?", "Earth", "string", []),
    ("What is the largest ocean on Earth?", "Pacific", "string", ["Pacific Ocean", "the Pacific"]),
    ("What is the tallest mountain above sea level?", "Everest", "string", ["Mount Everest"]),
    ("What is the longest river in the world?", "Nile", "string", ["the Nile", "Nile River"]),
    ("Which animal is known as the king of the jungle?", "lion", "string", ["the lion"]),
    ("What animal says 'meow'?", "cat", "string", ["a cat", "the cat"]),
    ("What animal guards houses and barks?", "dog", "string", ["a dog", "the dog"]),
    ("Which farm animal gives us milk?", "cow", "string", ["cows", "a cow", "dairy cow"]),
    ("What sweet food do bees make?", "honey", "string", []),
    ("How many days are there in a week?", "7", "numeric", []),
    ("How many days are there in a common year?", "365", "numeric", []),
    ("How many days are there in a leap year?", "366", "numeric", []),
    ("How many hours are there in one day?", "24", "numeric", []),
    ("How many minutes are there in one hour?", "60", "numeric", []),
    ("How many seconds are there in one minute?", "60", "numeric", []),
    ("How many months are there in a year?", "12", "numeric", []),
    ("How many sides does a triangle have?", "3", "numeric", []),
    ("How many sides does a square have?", "4", "numeric", []),
    ("How many corners does a rectangle have?", "4", "numeric", []),
    ("How many wheels does a car have?", "4", "numeric", []),
    ("How many legs does a spider have?", "8", "numeric", []),
    ("How many legs does a dog have?", "4", "numeric", []),
    ("How many fingers are on one human hand?", "5", "numeric", []),
    ("How many players from one soccer team are on the field?", "11", "numeric", []),
    ("At what temperature (°C) does water freeze?", "0", "numeric", []),
    ("At what temperature (°C) does water boil?", "100", "numeric", []),
    ("What is the common name for H2O?", "water", "string", []),
    ("Which gas do humans need to breathe in to stay alive?", "oxygen", "string", []),
    ("Which season comes right after winter?", "spring", "string", []),
    ("What is the opposite of hot?", "cold", "string", []),
    ("What is the opposite of big?", "small", "string", ["little"]),
    ("What does a lemon taste like?", "sour", "string", []),
    ("What does sugar taste like?", "sweet", "string", []),
    ("What meal do people eat in the morning?", "breakfast", "string", []),
    ("What is the opposite of day?", "night", "string", []),
    ("What device is commonly used to tell the time?", "clock", "string", ["a clock", "watch", "a watch"]),
    ("What do you use to write on paper: a pen or a plate?", "pen", "string", ["a pen"]),
    ("How many zeros are there in the number one thousand?", "3", "numeric", ["three"]),
    ("Which is bigger, an elephant or a mouse?", "elephant", "string", ["an elephant", "the elephant"]),
    ("What rises in the east every morning?", "sun", "string", ["the sun"]),
    # -- additional capitals --
    ("What is the capital city of Thailand?", "Bangkok", "string", []),
    ("What is the capital city of Russia?", "Moscow", "string", []),
    ("What is the capital city of India?", "New Delhi", "string", ["Delhi"]),
    ("What is the capital city of the United States?", "Washington, D.C.", "string",
     ["Washington", "Washington D.C.", "Washington, DC", "Washington DC"]),
    ("What is the capital city of the United Kingdom?", "London", "string", []),
    ("What is the capital city of South Korea?", "Seoul", "string", []),
    ("What is the capital city of Brazil?", "Brasilia", "string", ["Brasília"]),
    ("What is the capital city of Mexico?", "Mexico City", "string", []),
    ("What is the capital city of Portugal?", "Lisbon", "string", []),
    ("What is the capital city of Greece?", "Athens", "string", []),
    ("What is the capital city of Sweden?", "Stockholm", "string", []),
    ("What is the capital city of Norway?", "Oslo", "string", []),
    ("What is the capital city of Denmark?", "Copenhagen", "string", []),
    ("What is the capital city of Poland?", "Warsaw", "string", []),
    ("What is the capital city of Austria?", "Vienna", "string", []),
    ("What is the capital city of Switzerland?", "Bern", "string", ["Berne"]),
    ("What is the capital city of the Netherlands?", "Amsterdam", "string", []),
    ("What is the capital city of Belgium?", "Brussels", "string", []),
    ("What is the capital city of Ireland?", "Dublin", "string", []),
    ("What is the capital city of Indonesia?", "Jakarta", "string", []),
    ("What is the capital city of Vietnam?", "Hanoi", "string", []),
    ("What is the capital city of Turkey?", "Ankara", "string", []),
    # -- animals --
    ("What animal says 'moo'?", "cow", "string", ["a cow", "the cow"]),
    ("What animal says 'quack'?", "duck", "string", ["a duck", "the duck"]),
    ("What animal says 'baa'?", "sheep", "string", ["a sheep", "the sheep", "lamb"]),
    ("What animal says 'oink'?", "pig", "string", ["a pig", "the pig"]),
    ("What animal has a long trunk?", "elephant", "string", ["an elephant", "the elephant"]),
    ("What animal has long ears and hops?", "rabbit", "string", ["a rabbit", "the rabbit", "bunny"]),
    ("What animal produces wool?", "sheep", "string", ["a sheep", "the sheep"]),
    ("What is a baby cat called?", "kitten", "string", ["a kitten"]),
    ("What is a baby dog called?", "puppy", "string", ["a puppy"]),
    ("What do caterpillars turn into?", "butterfly", "string", ["butterflies", "a butterfly"]),
    # -- colors, body & senses --
    ("What color is an orange fruit?", "orange", "string", []),
    ("What color is a ripe strawberry?", "red", "string", []),
    ("What color is coal?", "black", "string", []),
    ("How many eyes does a human have?", "2", "numeric", ["two"]),
    ("How many ears does a human have?", "2", "numeric", ["two"]),
    ("How many arms does a human have?", "2", "numeric", ["two"]),
    ("What organ pumps blood through the body?", "heart", "string", ["the heart"]),
    ("What do humans use to see?", "eyes", "string", ["eye", "their eyes"]),
    # -- basics, time & opposites --
    ("How many letters are there in the English alphabet?", "26", "numeric", ["twenty-six"]),
    ("How many continents are there on Earth?", "7", "numeric", ["seven"]),
    ("How many colors are there in a rainbow?", "7", "numeric", ["seven"]),
    ("How many sides does a pentagon have?", "5", "numeric", ["five"]),
    ("How many sides does a hexagon have?", "6", "numeric", ["six"]),
    ("How many sides does an octagon have?", "8", "numeric", ["eight"]),
    ("How many players from one basketball team are on the court?", "5", "numeric", ["five"]),
    ("How many strings does a standard guitar have?", "6", "numeric", ["six"]),
    ("How many zeros are there in the number one hundred?", "2", "numeric", ["two"]),
    ("How many zeros are there in the number one million?", "6", "numeric", ["six"]),
    ("How many years are there in a decade?", "10", "numeric", ["ten"]),
    ("How many years are there in a century?", "100", "numeric", ["one hundred"]),
    ("What day of the week comes right after Monday?", "Tuesday", "string", []),
    ("What day of the week comes right after Friday?", "Saturday", "string", []),
    ("What is the opposite of cold?", "hot", "string", ["warm"]),
    ("What is the opposite of fast?", "slow", "string", []),
    ("What is the opposite of up?", "down", "string", []),
    ("What is the opposite of young?", "old", "string", []),
    ("What is the opposite of happy?", "sad", "string", []),
    ("What is the opposite of dark?", "light", "string", ["bright"]),
    ("What is frozen water called?", "ice", "string", []),
    ("Which star gives Earth light and heat?", "sun", "string", ["the sun", "the Sun", "Sun"]),
    ("What season is usually the coldest?", "winter", "string", []),
    ("What season is usually the hottest?", "summer", "string", []),
]

# parametric counting facts: (template, multiplier, max_n)
_COMMON_PARAMETRIC = [
    ("How many days are there in {n} weeks?", 7, 15),
    ("How many hours are there in {n} days?", 24, 10),
    ("How many minutes are there in {n} hours?", 60, 12),
    ("How many cents are there in {n} dollars?", 100, 12),
    ("How many eggs are there in {n} dozen eggs?", 12, 12),
    ("How many months are there in {n} years?", 12, 12),
    ("How many sides do {n} triangles have in total?", 3, 15),
    ("How many wheels do {n} bicycles have in total?", 2, 15),
    ("How many legs do {n} spiders have in total?", 8, 12),
    ("How many legs do {n} cats have in total?", 4, 12),
    ("How many fingers do {n} people have in total, counting both hands?", 10, 15),
    ("How many wheels do {n} tricycles have in total?", 3, 12),
    ("How many legs do {n} dogs have in total?", 4, 12),
    ("How many wheels do {n} cars have in total?", 4, 12),
    ("How many years are there in {n} decades?", 10, 10),
    ("How many halves are there in {n} whole pizzas?", 2, 10),
    ("How many minutes are there in {n} half-hours?", 30, 10),
]


def _common_pool(rng):
    items = []
    for q, a, t, alts in _COMMON_HANDWRITTEN:
        items.append({"category": "common_sense", "needs_tool": False,
                      "question": q, "expression": None, "answer": a,
                      "answer_type": t, "alternatives": alts})
    param = []
    for template, mult, max_n in _COMMON_PARAMETRIC:
        for n in range(2, max_n + 1):
            param.append((template, mult, n))
    rng.shuffle(param)
    for template, mult, n in param:
        items.append({"category": "common_sense", "needs_tool": False,
                      "question": template.format(n=n), "expression": None,
                      "answer": str(mult * n), "answer_type": "numeric",
                      "alternatives": []})
    return items


# --------------------------------------------------------------------------
# pool builders with dedup
# --------------------------------------------------------------------------

def _build_pool(rng, fn, count, *args, exclude=None):
    """Build a pool of ``count`` unique items.

    ``exclude`` is a shared registry of question strings that must not appear
    in the pool; every accepted question is added to it, so pools built with
    the same registry are pairwise disjoint (prevents train->eval leakage --
    magnitude ranges of hard_in vs hard_ood overlap for several kinds).
    """
    if exclude is None:
        exclude = set()
    pool, guard = [], 0
    while len(pool) < count and guard < count * 50:
        guard += 1
        item = fn(rng, *args) if args else fn(rng)
        if item["question"] in exclude:
            continue
        exclude.add(item["question"])
        pool.append(item)
    if len(pool) < count:
        raise RuntimeError(f"could not generate {count} unique items (got {len(pool)})")
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Global question registry: guarantees ZERO duplicate questions across all
    # four splits (train->eval leakage would invalidate the headline results).
    seen_global = set()

    # needed counts per category:            train, eval, ood, ratio20
    hard_in = _build_pool(rng, _hard_item, 350 + 24 + 40, False, exclude=seen_global)   # +slack
    hard_ood = _build_pool(rng, _hard_item, 60 + 20, True, exclude=seen_global)
    easy = _build_pool(rng, _easy_item, 180 + 50 + 30 + 60 + 40, exclude=seen_global)
    common = _common_pool(rng)
    common = [it for it in common if it["question"] not in seen_global]
    seen_global.update(it["question"] for it in common)
    rng.shuffle(common)
    need_common = 120 + 50 + 30 + 36
    if len(common) < need_common:
        raise RuntimeError(f"common pool too small: {len(common)} < {need_common}")

    rng.shuffle(hard_in); rng.shuffle(hard_ood); rng.shuffle(easy)

    hi, ho, ei, ci = 0, 0, 0, 0

    def take(pool, idx, k):
        chunk = pool[idx: idx + k]
        return chunk, idx + k

    splits = {"train": [], "eval": [], "eval_ood": [], "eval_ratio20": []}

    chunk, hi = take(hard_in, hi, 300); splits["train"] += chunk
    chunk, ei = take(easy, ei, 180);    splits["train"] += chunk
    chunk, ci = take(common, ci, 120);  splits["train"] += chunk

    chunk, hi = take(hard_in, hi, 50);  splits["eval"] += chunk
    chunk, ei = take(easy, ei, 50);     splits["eval"] += chunk
    chunk, ci = take(common, ci, 50);   splits["eval"] += chunk

    chunk, ho = take(hard_ood, ho, 60); splits["eval_ood"] += chunk
    chunk, ei = take(easy, ei, 30);     splits["eval_ood"] += chunk
    chunk, ci = take(common, ci, 30);   splits["eval_ood"] += chunk

    chunk, hi = take(hard_in, hi, 24);  splits["eval_ratio20"] += chunk
    chunk, ei = take(easy, ei, 60);     splits["eval_ratio20"] += chunk
    chunk, ci = take(common, ci, 36);   splits["eval_ratio20"] += chunk

    total = 0
    for split, rows in splits.items():
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            row["id"] = f"{split}-{i:04d}"
            row["split"] = split
        path = os.path.join(args.out_dir, f"{split}.jsonl")
        save_jsonl(path, rows)
        n_tool = sum(r["needs_tool"] for r in rows)
        total += len(rows)
        print(f"{path:42} {len(rows):4d} items  needs_tool={n_tool/len(rows):.0%}  "
              f"(hard={sum(r['category']=='hard_arith' for r in rows)}, "
              f"easy={sum(r['category']=='easy_arith' for r in rows)}, "
              f"common={sum(r['category']=='common_sense' for r in rows)})")
    print(f"total: {total} items")


if __name__ == "__main__":
    main()
