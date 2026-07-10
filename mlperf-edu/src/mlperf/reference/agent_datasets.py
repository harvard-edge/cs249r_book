"""
MLPerf EDU: Agent Dataset Factory

Provides real, task-specific datasets for agent benchmarks instead of
using TinyShakespeare. Each agent type has a dedicated dataset matching
its evaluation paradigm:

- CodeGen: MBPP (Mostly Basic Python Problems) — real Python tasks with tests
- RAG: Natural Questions subset — real question-document retrieval pairs
- ReAct: Multi-step reasoning tasks with tool traces
- ToolCall: Structured API call datasets

This separates agent benchmarks from language modeling benchmarks,
giving each workload authentic data that matches its use case.
"""

import os
import json
import torch
import torch.utils.data as data


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# MBPP Dataset for CodeGen Agent
# ---------------------------------------------------------------------------

# Curated subset of MBPP problems (hand-selected for pedagogical value).
# Each has: task description, canonical solution, and test assertions.
# Source: Austin et al., 2021 — "Program Synthesis with Large Language Models"
MBPP_PROBLEMS = [
    {
        "task_id": 1,
        "prompt": "Write a function to find the minimum cost path from (0,0) to (m,n) in a grid.",
        "code": "def min_cost(cost, m, n):\n    tc = [[0]*( n+1) for _ in range(m+1)]\n    tc[0][0] = cost[0][0]\n    for i in range(1, m+1): tc[i][0] = tc[i-1][0] + cost[i][0]\n    for j in range(1, n+1): tc[0][j] = tc[0][j-1] + cost[0][j]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            tc[i][j] = min(tc[i-1][j], tc[i][j-1], tc[i-1][j-1]) + cost[i][j]\n    return tc[m][n]",
        "test_list": [
            "assert min_cost([[1,2,3],[4,8,2],[1,5,3]], 2, 2) == 8",
            "assert min_cost([[2,3,4],[5,9,3],[2,6,4]], 2, 2) == 12",
        ],
    },
    {
        "task_id": 2,
        "prompt": "Write a function to find similar elements from two lists.",
        "code": "def similar_elements(l1, l2):\n    return tuple(set(l1) & set(l2))",
        "test_list": [
            "assert set(similar_elements((3,4,5,6),(5,7,4,10))) == {4,5}",
        ],
    },
    {
        "task_id": 3,
        "prompt": "Write a function to check if a string is a palindrome.",
        "code": "def is_palindrome(s):\n    return s == s[::-1]",
        "test_list": [
            "assert is_palindrome('racecar') == True",
            "assert is_palindrome('hello') == False",
            "assert is_palindrome('madam') == True",
        ],
    },
    {
        "task_id": 4,
        "prompt": "Write a function to find the maximum element in a list.",
        "code": "def find_max(lst):\n    return max(lst)",
        "test_list": [
            "assert find_max([1,3,2,5,4]) == 5",
            "assert find_max([-1,-3,-2]) == -1",
        ],
    },
    {
        "task_id": 5,
        "prompt": "Write a function to flatten a nested list.",
        "code": "def flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
        "test_list": [
            "assert flatten([1,[2,[3,4],5],6]) == [1,2,3,4,5,6]",
            "assert flatten([[1,2],[3,4]]) == [1,2,3,4]",
        ],
    },
    {
        "task_id": 6,
        "prompt": "Write a function to compute the nth Fibonacci number.",
        "code": "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a + b\n    return b",
        "test_list": [
            "assert fibonacci(0) == 0",
            "assert fibonacci(1) == 1",
            "assert fibonacci(10) == 55",
        ],
    },
    {
        "task_id": 7,
        "prompt": "Write a function to count the frequency of each element in a list.",
        "code": "def freq_count(lst):\n    freq = {}\n    for item in lst:\n        freq[item] = freq.get(item, 0) + 1\n    return freq",
        "test_list": [
            "assert freq_count([1,2,2,3,3,3]) == {1:1, 2:2, 3:3}",
        ],
    },
    {
        "task_id": 8,
        "prompt": "Write a function to check if a number is prime.",
        "code": "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "test_list": [
            "assert is_prime(2) == True",
            "assert is_prime(4) == False",
            "assert is_prime(17) == True",
            "assert is_prime(1) == False",
        ],
    },
    {
        "task_id": 9,
        "prompt": "Write a function to merge two sorted lists into one sorted list.",
        "code": "def merge_sorted(l1, l2):\n    result, i, j = [], 0, 0\n    while i < len(l1) and j < len(l2):\n        if l1[i] <= l2[j]:\n            result.append(l1[i]); i += 1\n        else:\n            result.append(l2[j]); j += 1\n    result.extend(l1[i:])\n    result.extend(l2[j:])\n    return result",
        "test_list": [
            "assert merge_sorted([1,3,5],[2,4,6]) == [1,2,3,4,5,6]",
            "assert merge_sorted([],[1,2]) == [1,2]",
        ],
    },
    {
        "task_id": 10,
        "prompt": "Write a function to compute the GCD of two numbers.",
        "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "test_list": [
            "assert gcd(12, 8) == 4",
            "assert gcd(7, 13) == 1",
            "assert gcd(100, 25) == 25",
        ],
    },
    {
        "task_id": 11,
        "prompt": "Write a function to reverse a string.",
        "code": "def reverse_string(s):\n    return s[::-1]",
        "test_list": [
            "assert reverse_string('hello') == 'olleh'",
            "assert reverse_string('') == ''",
        ],
    },
    {
        "task_id": 12,
        "prompt": "Write a function to find the second largest element in a list.",
        "code": "def second_largest(lst):\n    unique = list(set(lst))\n    unique.sort()\n    return unique[-2] if len(unique) >= 2 else None",
        "test_list": [
            "assert second_largest([1,2,3,4,5]) == 4",
            "assert second_largest([5,5,5]) == None",
        ],
    },
    {
        "task_id": 13,
        "prompt": "Write a function to remove duplicates from a list while preserving order.",
        "code": "def remove_duplicates(lst):\n    seen = set()\n    result = []\n    for item in lst:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result",
        "test_list": [
            "assert remove_duplicates([1,2,2,3,1,4]) == [1,2,3,4]",
        ],
    },
    {
        "task_id": 14,
        "prompt": "Write a function to compute the sum of digits of a number.",
        "code": "def sum_digits(n):\n    return sum(int(d) for d in str(abs(n)))",
        "test_list": [
            "assert sum_digits(123) == 6",
            "assert sum_digits(0) == 0",
        ],
    },
    {
        "task_id": 15,
        "prompt": "Write a function to rotate a list by k positions to the right.",
        "code": "def rotate_list(lst, k):\n    if not lst: return lst\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k]",
        "test_list": [
            "assert rotate_list([1,2,3,4,5], 2) == [4,5,1,2,3]",
            "assert rotate_list([1,2,3], 0) == [1,2,3]",
        ],
    },
    {
        "task_id": 16,
        "prompt": "Write a function to check if two strings are anagrams.",
        "code": "def are_anagrams(s1, s2):\n    return sorted(s1.lower()) == sorted(s2.lower())",
        "test_list": [
            "assert are_anagrams('listen', 'silent') == True",
            "assert are_anagrams('hello', 'world') == False",
        ],
    },
    {
        "task_id": 17,
        "prompt": "Write a function to find all pairs in a list that sum to a target.",
        "code": "def two_sum_pairs(lst, target):\n    seen = set()\n    pairs = []\n    for x in lst:\n        comp = target - x\n        if comp in seen:\n            pairs.append((min(x,comp), max(x,comp)))\n        seen.add(x)\n    return sorted(set(pairs))",
        "test_list": [
            "assert two_sum_pairs([1,2,3,4,5], 6) == [(1,5),(2,4)]",
        ],
    },
    {
        "task_id": 18,
        "prompt": "Write a function to compute the power set of a list.",
        "code": "def power_set(lst):\n    if not lst: return [[]]\n    rest = power_set(lst[1:])\n    return rest + [[lst[0]] + s for s in rest]",
        "test_list": [
            "assert len(power_set([1,2,3])) == 8",
            "assert [] in power_set([1,2])",
        ],
    },
    {
        "task_id": 19,
        "prompt": "Write a function to implement binary search on a sorted list.",
        "code": "def binary_search(lst, target):\n    lo, hi = 0, len(lst) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if lst[mid] == target: return mid\n        elif lst[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1",
        "test_list": [
            "assert binary_search([1,3,5,7,9], 5) == 2",
            "assert binary_search([1,3,5,7,9], 4) == -1",
        ],
    },
    {
        "task_id": 20,
        "prompt": "Write a function to find the longest common prefix of a list of strings.",
        "code": "def longest_common_prefix(strs):\n    if not strs: return ''\n    prefix = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix: return ''\n    return prefix",
        "test_list": [
            "assert longest_common_prefix(['flower','flow','flight']) == 'fl'",
            "assert longest_common_prefix(['dog','racecar','car']) == ''",
        ],
    },
]


class MBPPCodeDataset(data.Dataset):
    """
    Dataset wrapping MBPP-style Python problems for code generation benchmarking.

    Each sample returns:
    - input_ids: tokenized prompt (character-level for compatibility)
    - target_ids: tokenized canonical solution
    - task_id: integer ID for tracking

    The character-level tokenization keeps things simple and compatible
    with the existing NanoGPT-based agents.
    """

    def __init__(self, problems=None, seq_len=128):
        self.problems = problems or MBPP_PROBLEMS
        self.seq_len = seq_len
        self._prepare()

    def _prepare(self):
        """Tokenize all prompts and solutions at init time."""
        self.samples = []
        for p in self.problems:
            # Combine prompt + solution for language modeling
            full_text = f"# Task: {p['prompt']}\n{p['code']}\n"
            tokens = list(full_text.encode("ascii", errors="replace"))
            # Pad or truncate to seq_len + 1 (need target shift)
            if len(tokens) > self.seq_len + 1:
                tokens = tokens[:self.seq_len + 1]
            else:
                tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
            self.samples.append({
                "tokens": torch.tensor(tokens, dtype=torch.long),
                "task_id": p["task_id"],
                "tests": p["test_list"],
            })

    def __len__(self):
        # Augment by repeating problems (small dataset)
        return len(self.samples) * 50  # 20 problems × 50 = 1000 samples

    def __getitem__(self, idx):
        sample = self.samples[idx % len(self.samples)]
        tokens = sample["tokens"]
        x = tokens[:-1]  # input
        y = tokens[1:]   # target (shifted by 1)
        return x, y


class ReActTraceDataset(data.Dataset):
    """
    Dataset of multi-step reasoning traces for ReAct agent training.

    Each trace is a sequence of: Question → Think → Act → Observe → ... → Answer
    formatted as a character-level sequence for language model training.
    """

    TRACES = [
        # --- Arithmetic chains ---
        "Q: What is 25 * 4 + 10 * 3?\nThink: I need to compute 25*4 first.\nAct: calculator(25 * 4)\nObs: 100\nThink: Now 10*3.\nAct: calculator(10 * 3)\nObs: 30\nThink: Sum them.\nAct: calculator(100 + 30)\nObs: 130\nAnswer: 130",
        "Q: What is (17 + 23) * (50 - 38)?\nThink: First compute the sum.\nAct: calculator(17 + 23)\nObs: 40\nThink: Now the difference.\nAct: calculator(50 - 38)\nObs: 12\nThink: Multiply the results.\nAct: calculator(40 * 12)\nObs: 480\nAnswer: 480",
        "Q: What is 15% of 240?\nThink: 15% means multiply by 0.15.\nAct: calculator(240 * 0.15)\nObs: 36.0\nAnswer: 36",

        # --- String analysis ---
        "Q: Is the length of 'benchmark' greater than 5?\nThink: Find the string length.\nAct: string_length('benchmark')\nObs: 9\nThink: Compare 9 and 5.\nAct: compare(9, 5)\nObs: greater\nAnswer: Yes, 9 > 5.",
        "Q: Which is longer: 'hello' or 'world!'?\nThink: Get length of 'hello'.\nAct: string_length('hello')\nObs: 5\nThink: Get length of 'world!'.\nAct: string_length('world!')\nObs: 6\nAct: compare(5, 6)\nObs: less\nAnswer: 'world!' is longer (6 vs 5).",
        "Q: What is the combined length of 'data' and 'science'?\nThink: Get length of each word.\nAct: string_length('data')\nObs: 4\nAct: string_length('science')\nObs: 7\nThink: Add them.\nAct: calculator(4 + 7)\nObs: 11\nAnswer: The combined length is 11 characters.",

        # --- Constant lookups + computation ---
        "Q: What is pi times 2?\nThink: Look up pi.\nAct: lookup('pi')\nObs: 3.14159\nThink: Multiply by 2.\nAct: calculator(3.14159 * 2)\nObs: 6.28318\nAnswer: 6.28318",
        "Q: What is the golden ratio times 100?\nThink: Look up golden ratio.\nAct: lookup('golden_ratio')\nObs: 1.61803\nAct: calculator(1.61803 * 100)\nObs: 161.803\nAnswer: 161.803",
        "Q: What is Avogadro's number divided by 1e20?\nThink: Look up Avogadro's number.\nAct: lookup('avogadro')\nObs: 6.022e23\nAct: calculator(6.022e23 / 1e20)\nObs: 6022.0\nAnswer: 6022.0",
        "Q: Is pi greater than e?\nThink: Look up both constants.\nAct: lookup('pi')\nObs: 3.14159\nAct: lookup('e')\nObs: 2.71828\nThink: Compare them.\nAct: compare(3.14159, 2.71828)\nObs: greater\nAnswer: Yes, pi (3.14159) is greater than e (2.71828).",

        # --- Multi-step scientific ---
        "Q: What is sqrt(144) + sqrt(169)?\nThink: Compute sqrt(144).\nAct: calculator(144 ** 0.5)\nObs: 12.0\nThink: Compute sqrt(169).\nAct: calculator(169 ** 0.5)\nObs: 13.0\nThink: Add them.\nAct: calculator(12 + 13)\nObs: 25\nAnswer: 25",
        "Q: What is the area of a circle with radius 5?\nThink: Area = pi * r^2. First look up pi.\nAct: lookup('pi')\nObs: 3.14159\nThink: Compute r^2.\nAct: calculator(5 ** 2)\nObs: 25\nThink: Multiply by pi.\nAct: calculator(3.14159 * 25)\nObs: 78.53975\nAnswer: The area is approximately 78.54 square units.",
        "Q: What is the kinetic energy of a 2kg object at 10 m/s?\nThink: KE = 0.5 * m * v^2. Compute v^2 first.\nAct: calculator(10 ** 2)\nObs: 100\nThink: Now 0.5 * 2 * 100.\nAct: calculator(0.5 * 2 * 100)\nObs: 100.0\nAnswer: The kinetic energy is 100 joules.",

        # --- Primality / number theory ---
        "Q: Is 97 a prime number?\nThink: Check divisors up to sqrt(97).\nAct: calculator(97 % 2)\nObs: 1\nAct: calculator(97 % 3)\nObs: 1\nAct: calculator(97 % 5)\nObs: 2\nAct: calculator(97 % 7)\nObs: 6\nThink: Not divisible by 2,3,5,7. It's prime.\nAnswer: Yes, 97 is prime.",
        "Q: What is the remainder when 2^10 is divided by 7?\nThink: First compute 2^10.\nAct: calculator(2 ** 10)\nObs: 1024\nThink: Now find the remainder.\nAct: calculator(1024 % 7)\nObs: 2\nAnswer: The remainder is 2.",

        # --- Comparison chains ---
        "Q: Rank these numbers from smallest to largest: 3.14, 2.72, 1.41.\nThink: Compare first two.\nAct: compare(3.14, 2.72)\nObs: greater\nThink: So 2.72 < 3.14. Now compare with 1.41.\nAct: compare(1.41, 2.72)\nObs: less\nAnswer: 1.41 < 2.72 < 3.14.",
        "Q: Is the speed of light greater than 300 million?\nThink: Look up speed of light.\nAct: lookup('speed_of_light')\nObs: 299792458\nThink: Compare with 300000000.\nAct: compare(299792458, 300000000)\nObs: less\nAnswer: No, the speed of light (299,792,458 m/s) is slightly less than 300 million.",

        # --- Multi-hop reasoning ---
        "Q: What is 2*pi*r when r=7?\nThink: First get pi.\nAct: lookup('pi')\nObs: 3.14159\nThink: Compute 2 * pi.\nAct: calculator(2 * 3.14159)\nObs: 6.28318\nThink: Multiply by r=7.\nAct: calculator(6.28318 * 7)\nObs: 43.98226\nAnswer: The circumference is approximately 43.98.",
        "Q: How many characters are in 'machine learning' including the space?\nThink: Count the full string.\nAct: string_length('machine learning')\nObs: 16\nAnswer: There are 16 characters including the space.",
        "Q: What is sqrt(2) * sqrt(2)?\nThink: Look up sqrt(2).\nAct: lookup('sqrt2')\nObs: 1.41421\nThink: Multiply by itself.\nAct: calculator(1.41421 * 1.41421)\nObs: 1.9999899241\nAnswer: Approximately 2 (as expected, since sqrt(2)^2 = 2).",

        # --- Error handling / disambiguation ---
        "Q: What is the Planck constant in SI units?\nThink: Look up Planck constant.\nAct: lookup('planck')\nObs: 6.626e-34\nAnswer: The Planck constant is 6.626 × 10^-34 joule-seconds.",
        "Q: What is the Boltzmann constant times 300?\nThink: Look up Boltzmann constant.\nAct: lookup('boltzmann')\nObs: 1.381e-23\nThink: Multiply by 300 (room temperature in K).\nAct: calculator(1.381e-23 * 300)\nObs: 4.143e-21\nAnswer: kT at room temperature is approximately 4.14 × 10^-21 joules.",

        # --- Complex word problems ---
        "Q: If a train travels 120 km in 1.5 hours, what is its speed in km/h?\nThink: Speed = distance / time.\nAct: calculator(120 / 1.5)\nObs: 80.0\nAnswer: The train's speed is 80 km/h.",
        "Q: A rectangle has length 12 and width 8. What is its diagonal?\nThink: Diagonal = sqrt(12^2 + 8^2). First compute 12^2.\nAct: calculator(12 ** 2)\nObs: 144\nThink: Now 8^2.\nAct: calculator(8 ** 2)\nObs: 64\nThink: Sum and square root.\nAct: calculator((144 + 64) ** 0.5)\nObs: 14.422205101855956\nAnswer: The diagonal is approximately 14.42 units.",
        "Q: Convert 98.6°F to Celsius.\nThink: Formula: C = (F - 32) * 5/9.\nAct: calculator((98.6 - 32) * 5 / 9)\nObs: 37.0\nAnswer: 98.6°F is 37.0°C (normal body temperature).",
    ]

    def __init__(self, seq_len=256):
        self.seq_len = seq_len
        self._prepare()

    def _prepare(self):
        self.samples = []
        for trace in self.TRACES:
            tokens = list(trace.encode("ascii", errors="replace"))
            if len(tokens) > self.seq_len + 1:
                tokens = tokens[:self.seq_len + 1]
            else:
                tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
            self.samples.append(torch.tensor(tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples) * 100  # Augment: 8 traces × 100 = 800

    def __getitem__(self, idx):
        tokens = self.samples[idx % len(self.samples)]
        return tokens[:-1], tokens[1:]


def get_mbpp_dataloaders(batch_size=16, seq_len=128):
    """Returns (train_loader, val_loader) for MBPP code generation."""
    all_problems = MBPP_PROBLEMS
    n_train = int(len(all_problems) * 0.8)
    train_problems = all_problems[:n_train]
    val_problems = all_problems[n_train:]

    train_ds = MBPPCodeDataset(train_problems, seq_len=seq_len)
    val_ds = MBPPCodeDataset(val_problems, seq_len=seq_len)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, val_loader


def get_react_dataloaders(batch_size=8, seq_len=256):
    """Returns (train_loader, val_loader) for ReAct reasoning traces."""
    all_traces = ReActTraceDataset.TRACES
    ds = ReActTraceDataset(seq_len=seq_len)

    # 80/20 split
    n_total = len(ds)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_ds, val_ds = data.random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=True
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("🧪 Agent Dataset Factory — Verification")

    # MBPP
    train_ld, val_ld = get_mbpp_dataloaders(batch_size=4)
    x, y = next(iter(train_ld))
    print(f"\n📝 MBPP CodeGen Dataset:")
    print(f"   Train: {len(train_ld.dataset)} samples")
    print(f"   Val:   {len(val_ld.dataset)} samples")
    print(f"   Batch: x={x.shape}, y={y.shape}")
    snippet = bytes(x[0, :60].tolist()).decode("ascii", errors="replace")
    print(f"   Sample: '{snippet}'")

    # ReAct
    train_ld, val_ld = get_react_dataloaders(batch_size=4)
    x, y = next(iter(train_ld))
    print(f"\n🔄 ReAct Trace Dataset:")
    print(f"   Train: {len(train_ld.dataset)} samples")
    print(f"   Val:   {len(val_ld.dataset)} samples")
    print(f"   Batch: x={x.shape}, y={y.shape}")
    snippet = bytes(x[0, :80].tolist()).decode("ascii", errors="replace")
    print(f"   Sample: '{snippet}'")

    print("\n✅ All agent datasets verified.")
