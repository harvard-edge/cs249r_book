"""TinyPy Dataset Generator.

Generates the official TinyPy educational dataset for TinyTorch
code completion and language modeling experiments.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass
class Snippet:
    """A single Python code snippet in the dataset."""

    name: str
    category: str
    code: str


def get_snippets() -> list[Snippet]:
    """Return the complete collection of curated Python snippets."""
    snippets: list[Snippet] = []

    # =========================================================================
    # =========================================================================
    # Category 0: Core Algorithmic Building Blocks
    # =========================================================================

    core_blocks = [
        (
            "add_basic",
            '''def add(a, b):
    return a + b
''',
        ),
        (
            "subtract_basic",
            '''def subtract(a, b):
    return a - b
''',
        ),
        (
            "multiply_basic",
            '''def multiply(a, b):
    return a * b
''',
        ),
        (
            "divide_basic",
            '''def divide(a, b):
    if b == 0:
        return 0
    return a / b
''',
        ),
        (
            "relu_basic",
            '''def relu(x):
    if x > 0:
        return x
    return 0
''',
        ),
        (
            "factorial_basic",
            '''def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
''',
        ),
        (
            "is_prime_basic",
            '''def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
        ),
        (
            "linear_basic",
            '''class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x
''',
        ),
        (
            "mlp_basic",
            '''class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2.forward(relu(self.fc1.forward(x)))
''',
        ),
    ]

    for repeat_idx in range(6):
        for name, code in core_blocks:
            snippets.append(
                Snippet(
                    name=f"{name}_{repeat_idx}",
                    category="Core Algorithmic Building Blocks",
                    code=code,
                )
            )

    # =========================================================================
    # Category 1: Math and Number Theory
    # =========================================================================

    snippets.append(
        Snippet(
            name="add",
            category="Math and Number Theory",
            code='''def add(a: float, b: float) -> float:
    """Compute the arithmetic sum of two numbers.

    Args:
        a: First scalar operand.
        b: Second scalar operand.

    Returns:
        Sum of a and b.
    """
    return a + b
''',
        )
    )

    snippets.append(
        Snippet(
            name="subtract",
            category="Math and Number Theory",
            code='''def subtract(a: float, b: float) -> float:
    """Compute the difference between two numbers.

    Args:
        a: Minuend value.
        b: Subtrahend value.

    Returns:
        Difference of a and b.
    """
    return a - b
''',
        )
    )

    snippets.append(
        Snippet(
            name="multiply",
            category="Math and Number Theory",
            code='''def multiply(a: float, b: float) -> float:
    """Compute the product of two numbers.

    Args:
        a: Multiplicand scalar value.
        b: Multiplier scalar value.

    Returns:
        Product of a and b.
    """
    return a * b
''',
        )
    )

    snippets.append(
        Snippet(
            name="divide",
            category="Math and Number Theory",
            code='''def divide(a: float, b: float) -> float:
    """Compute the quotient of two numbers.

    Args:
        a: Dividend value.
        b: Divisor value.

    Returns:
        Quotient of a divided by b.

    Raises:
        ZeroDivisionError: If b is zero.
    """
    if b == 0.0:
        raise ZeroDivisionError("Divisor cannot be zero.")
    return a / b
''',
        )
    )

    snippets.append(
        Snippet(
            name="is_even",
            category="Math and Number Theory",
            code='''def is_even(n: int) -> bool:
    """Determine whether an integer is even.

    Args:
        n: Integer number to inspect.

    Returns:
        True if n is divisible by two, False otherwise.
    """
    return n % 2 == 0
''',
        )
    )

    snippets.append(
        Snippet(
            name="is_odd",
            category="Math and Number Theory",
            code='''def is_odd(n: int) -> bool:
    """Determine whether an integer is odd.

    Args:
        n: Integer number to inspect.

    Returns:
        True if n is not divisible by two, False otherwise.
    """
    return n % 2 != 0
''',
        )
    )

    snippets.append(
        Snippet(
            name="absolute_value",
            category="Math and Number Theory",
            code='''def absolute_value(x: float) -> float:
    """Compute the non-negative magnitude of a real number.

    Args:
        x: Real number input.

    Returns:
        Non-negative absolute value of x.
    """
    if x < 0.0:
        return -x
    return x
''',
        )
    )

    snippets.append(
        Snippet(
            name="factorial",
            category="Math and Number Theory",
            code='''def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer.

    Args:
        n: Non-negative integer.

    Returns:
        Factorial product n! as an integer.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is undefined for negative integers.")
    result: int = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="fibonacci",
            category="Math and Number Theory",
            code='''def fibonacci(n: int) -> int:
    """Compute the n-th Fibonacci number iteratively.

    Args:
        n: Zero-indexed position in Fibonacci sequence.

    Returns:
        The n-th Fibonacci integer.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Index cannot be negative.")
    if n == 0:
        return 0
    if n == 1:
        return 1
    prev: int = 0
    curr: int = 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
''',
        )
    )

    snippets.append(
        Snippet(
            name="is_prime",
            category="Math and Number Theory",
            code='''def is_prime(n: int) -> bool:
    """Check if an integer is a prime number.

    Args:
        n: Candidate integer to test.

    Returns:
        True if n is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    divisor: int = 5
    while divisor * divisor <= n:
        if n % divisor == 0 or n % (divisor + 2) == 0:
            return False
        divisor += 6
    return True
''',
        )
    )

    snippets.append(
        Snippet(
            name="gcd",
            category="Math and Number Theory",
            code='''def gcd(a: int, b: int) -> int:
    """Compute the greatest common divisor using Euclidean algorithm.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Greatest common divisor of a and b.
    """
    x: int = abs(a)
    y: int = abs(b)
    while y != 0:
        x, y = y, x % y
    return x
''',
        )
    )

    snippets.append(
        Snippet(
            name="lcm",
            category="Math and Number Theory",
            code='''def lcm(a: int, b: int) -> int:
    """Compute the least common multiple of two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Least common multiple of a and b.
    """
    if a == 0 or b == 0:
        return 0
    x: int = abs(a)
    y: int = abs(b)
    temp_x: int = x
    temp_y: int = y
    while temp_y != 0:
        temp_x, temp_y = temp_y, temp_x % temp_y
    common_divisor: int = temp_x
    return (x * y) // common_divisor
''',
        )
    )

    snippets.append(
        Snippet(
            name="power",
            category="Math and Number Theory",
            code='''def power(base: float, exp: int) -> float:
    """Calculate base raised to an integer exponent by binary squaring.

    Args:
        base: Real base number.
        exp: Integer power.

    Returns:
        Result of base raised to exp.
    """
    if exp == 0:
        return 1.0
    if exp < 0:
        base = 1.0 / base
        exp = -exp
    result: float = 1.0
    current_factor: float = base
    while exp > 0:
        if exp % 2 == 1:
            result *= current_factor
        current_factor *= current_factor
        exp //= 2
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="clamp",
            category="Math and Number Theory",
            code='''def clamp(val: float, low: float, high: float) -> float:
    """Constrain a value to lie within a specified closed interval.

    Args:
        val: Input scalar value.
        low: Minimum acceptable threshold.
        high: Maximum acceptable threshold.

    Returns:
        Clamped value bounded by low and high.
    """
    if val < low:
        return low
    if val > high:
        return high
    return val
''',
        )
    )

    snippets.append(
        Snippet(
            name="clip",
            category="Math and Number Theory",
            code='''def clip(val: float, min_val: float, max_val: float) -> float:
    """Restrict a value within lower and upper bounds.

    Args:
        val: Numerical value to clip.
        min_val: Lower numerical limit.
        max_val: Upper numerical limit.

    Returns:
        Value restricted to the specified limits.
    """
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val
''',
        )
    )

    snippets.append(
        Snippet(
            name="mean",
            category="Math and Number Theory",
            code='''def mean(numbers: list[float]) -> float:
    """Compute the arithmetic average of a sequence of numbers.

    Args:
        numbers: List of floating point numbers.

    Returns:
        Arithmetic mean value.

    Raises:
        ValueError: If the input list is empty.
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list.")
    total: float = sum(numbers)
    return total / len(numbers)
''',
        )
    )

    snippets.append(
        Snippet(
            name="variance",
            category="Math and Number Theory",
            code='''def variance(numbers: list[float]) -> float:
    """Compute the sample variance of a collection of values.

    Args:
        numbers: List of numerical measurements.

    Returns:
        Sample variance of the data.

    Raises:
        ValueError: If fewer than two data points are provided.
    """
    n: int = len(numbers)
    if n < 2:
        raise ValueError("Sample variance requires at least two points.")
    avg: float = sum(numbers) / n
    sum_squared_diffs: float = sum((x - avg) ** 2 for x in numbers)
    return sum_squared_diffs / (n - 1)
''',
        )
    )

    snippets.append(
        Snippet(
            name="standard_deviation",
            category="Math and Number Theory",
            code='''def standard_deviation(numbers: list[float]) -> float:
    """Compute the sample standard deviation of numerical data.

    Args:
        numbers: List of observations.

    Returns:
        Sample standard deviation.

    Raises:
        ValueError: If list has fewer than two elements.
    """
    n: int = len(numbers)
    if n < 2:
        raise ValueError("Standard deviation requires at least two values.")
    avg: float = sum(numbers) / n
    sum_sq: float = sum((x - avg) ** 2 for x in numbers)
    return math.sqrt(sum_sq / (n - 1))
''',
        )
    )

    snippets.append(
        Snippet(
            name="median",
            category="Math and Number Theory",
            code='''def median(numbers: list[float]) -> float:
    """Find the median value of a sequence of numbers.

    Args:
        numbers: List of numeric values.

    Returns:
        Median value of the collection.

    Raises:
        ValueError: If the sequence is empty.
    """
    if not numbers:
        raise ValueError("Cannot compute median of empty collection.")
    sorted_vals: list[float] = sorted(numbers)
    count: int = len(sorted_vals)
    midpoint: int = count // 2
    if count % 2 == 1:
        return sorted_vals[midpoint]
    return (sorted_vals[midpoint - 1] + sorted_vals[midpoint]) / 2.0
''',
        )
    )

    snippets.append(
        Snippet(
            name="is_perfect_square",
            category="Math and Number Theory",
            code='''def is_perfect_square(n: int) -> bool:
    """Determine whether an integer is a perfect square.

    Args:
        n: Integer number to check.

    Returns:
        True if n is the square of an integer, False otherwise.
    """
    if n < 0:
        return False
    root: int = int(math.isqrt(n))
    return root * root == n
''',
        )
    )

    snippets.append(
        Snippet(
            name="degrees_to_radians",
            category="Math and Number Theory",
            code='''def degrees_to_radians(degrees: float) -> float:
    """Convert an angular measure from degrees to radians.

    Args:
        degrees: Angle in degrees.

    Returns:
        Equivalent angle in radians.
    """
    return degrees * (math.pi / 180.0)
''',
        )
    )

    snippets.append(
        Snippet(
            name="radians_to_degrees",
            category="Math and Number Theory",
            code='''def radians_to_degrees(radians: float) -> float:
    """Convert an angular measure from radians to degrees.

    Args:
        radians: Angle in radians.

    Returns:
        Equivalent angle in degrees.
    """
    return radians * (180.0 / math.pi)
''',
        )
    )

    snippets.append(
        Snippet(
            name="hypotenuse",
            category="Math and Number Theory",
            code='''def hypotenuse(a: float, b: float) -> float:
    """Calculate the length of the hypotenuse in a right triangle.

    Args:
        a: Length of the first perpendicular side.
        b: Length of the second perpendicular side.

    Returns:
        Length of the hypotenuse side.
    """
    return math.sqrt(a * a + b * b)
''',
        )
    )

    snippets.append(
        Snippet(
            name="harmonic_mean",
            category="Math and Number Theory",
            code='''def harmonic_mean(numbers: list[float]) -> float:
    """Calculate the harmonic mean of positive numbers.

    Args:
        numbers: List of strictly positive real numbers.

    Returns:
        Harmonic mean value.

    Raises:
        ValueError: If list is empty or contains non-positive numbers.
    """
    if not numbers:
        raise ValueError("List cannot be empty.")
    reciprocal_sum: float = 0.0
    for val in numbers:
        if val <= 0.0:
            raise ValueError("All numbers must be strictly positive.")
        reciprocal_sum += 1.0 / val
    return len(numbers) / reciprocal_sum
''',
        )
    )

    snippets.append(
        Snippet(
            name="geometric_mean",
            category="Math and Number Theory",
            code='''def geometric_mean(numbers: list[float]) -> float:
    """Calculate the geometric mean of positive numbers.

    Args:
        numbers: List of strictly positive real values.

    Returns:
        Geometric mean of the values.

    Raises:
        ValueError: If input is empty or has non-positive numbers.
    """
    if not numbers:
        raise ValueError("List cannot be empty.")
    log_sum: float = 0.0
    for val in numbers:
        if val <= 0.0:
            raise ValueError("Numbers must be positive.")
        log_sum += math.log(val)
    return math.exp(log_sum / len(numbers))
''',
        )
    )

    snippets.append(
        Snippet(
            name="digital_root",
            category="Math and Number Theory",
            code='''def digital_root(n: int) -> int:
    """Compute the recursive single-digit sum of an integer.

    Args:
        n: Non-negative integer.

    Returns:
        Single-digit root between 0 and 9.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Integer must be non-negative.")
    if n == 0:
        return 0
    remainder: int = n % 9
    return 9 if remainder == 0 else remainder
''',
        )
    )

    snippets.append(
        Snippet(
            name="combinations_count",
            category="Math and Number Theory",
            code='''def combinations_count(n: int, k: int) -> int:
    """Compute the binomial coefficient n choose k.

    Args:
        n: Total number of items.
        k: Number of chosen items.

    Returns:
        Number of distinct combinations.

    Raises:
        ValueError: If parameters are invalid.
    """
    if n < 0 or k < 0:
        raise ValueError("Values must be non-negative.")
    if k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    numerator: int = 1
    denominator: int = 1
    for i in range(1, k + 1):
        numerator *= n - (k - i)
        denominator *= i
    return numerator // denominator
''',
        )
    )

    snippets.append(
        Snippet(
            name="permutations_count",
            category="Math and Number Theory",
            code='''def permutations_count(n: int, k: int) -> int:
    """Compute the number of ordered arrangements of k items from n.

    Args:
        n: Size of the total population.
        k: Size of the sample arrangement.

    Returns:
        Number of distinct ordered permutations.

    Raises:
        ValueError: If inputs are invalid.
    """
    if n < 0 or k < 0:
        raise ValueError("Inputs must be non-negative.")
    if k > n:
        return 0
    result: int = 1
    for i in range(n - k + 1, n + 1):
        result *= i
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="sign",
            category="Math and Number Theory",
            code='''def sign(x: float) -> int:
    """Extract the sign indicator of a real number.

    Args:
        x: Real number to test.

    Returns:
        1 if positive, -1 if negative, 0 if zero.
    """
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0
''',
        )
    )

    snippets.append(
        Snippet(
            name="sum_of_squares",
            category="Math and Number Theory",
            code='''def sum_of_squares(numbers: list[float]) -> float:
    """Compute the total sum of squared values in a sequence.

    Args:
        numbers: List of floating point values.

    Returns:
        Sum of each element multiplied by itself.
    """
    return sum(x * x for x in numbers)
''',
        )
    )

    # =========================================================================
    # Category 2: Classic Algorithms and Data Structures
    # =========================================================================

    snippets.append(
        Snippet(
            name="linear_search",
            category="Classic Algorithms and Data Structures",
            code='''def linear_search(arr: list[int], target: int) -> int:
    """Search sequentially for a target integer within an array.

    Args:
        arr: List of integers to search.
        target: Target integer sought.

    Returns:
        Zero-based index of target if found, otherwise -1.
    """
    for index, val in enumerate(arr):
        if val == target:
            return index
    return -1
''',
        )
    )

    snippets.append(
        Snippet(
            name="binary_search",
            category="Classic Algorithms and Data Structures",
            code='''def binary_search(arr: list[int], target: int) -> int:
    """Perform logarithmic search on a sorted integer list.

    Args:
        arr: Sorted array of integers in non-decreasing order.
        target: Value to locate.

    Returns:
        Index of the target if present, otherwise -1.
    """
    low: int = 0
    high: int = len(arr) - 1
    while low <= high:
        mid: int = low + (high - low) // 2
        mid_val: int = arr[mid]
        if mid_val == target:
            return mid
        if mid_val < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
        )
    )

    snippets.append(
        Snippet(
            name="bubble_sort",
            category="Classic Algorithms and Data Structures",
            code='''def bubble_sort(arr: list[int]) -> list[int]:
    """Sort an array using the bubble sort algorithm.

    Args:
        arr: List of integers to sort.

    Returns:
        New list containing elements in sorted order.
    """
    result: list[int] = list(arr)
    n: int = len(result)
    for i in range(n):
        swapped: bool = False
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True
        if not swapped:
            break
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="selection_sort",
            category="Classic Algorithms and Data Structures",
            code='''def selection_sort(arr: list[int]) -> list[int]:
    """Sort a list of integers using selection sort.

    Args:
        arr: Unsorted integer list.

    Returns:
        Sorted integer list.
    """
    result: list[int] = list(arr)
    n: int = len(result)
    for i in range(n):
        min_idx: int = i
        for j in range(i + 1, n):
            if result[j] < result[min_idx]:
                min_idx = j
        if min_idx != i:
            result[i], result[min_idx] = result[min_idx], result[i]
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="insertion_sort",
            category="Classic Algorithms and Data Structures",
            code='''def insertion_sort(arr: list[int]) -> list[int]:
    """Sort an array using the insertion sort algorithm.

    Args:
        arr: List of integers to sort.

    Returns:
        New sorted list of integers.
    """
    result: list[int] = list(arr)
    for i in range(1, len(result)):
        key: int = result[i]
        j: int = i - 1
        while j >= 0 and result[j] > key:
            result[j + 1] = result[j]
            j -= 1
        result[j + 1] = key
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="merge_sort",
            category="Classic Algorithms and Data Structures",
            code='''def merge_sort(arr: list[int]) -> list[int]:
    """Sort an array of integers using divide and conquer merge sort.

    Args:
        arr: Integer list to sort.

    Returns:
        Sorted list containing all elements.
    """
    if len(arr) <= 1:
        return list(arr)

    mid: int = len(arr) // 2
    left: list[int] = merge_sort(arr[:mid])
    right: list[int] = merge_sort(arr[mid:])

    merged: list[int] = []
    i: int = 0
    j: int = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged
''',
        )
    )

    snippets.append(
        Snippet(
            name="quicksort",
            category="Classic Algorithms and Data Structures",
            code='''def quicksort(arr: list[int]) -> list[int]:
    """Sort an array recursively using Hoare divide and conquer logic.

    Args:
        arr: Unsorted list of integers.

    Returns:
        Sorted list in non-decreasing order.
    """
    if len(arr) <= 1:
        return list(arr)
    pivot: int = arr[len(arr) // 2]
    less: list[int] = [x for x in arr if x < pivot]
    equal: list[int] = [x for x in arr if x == pivot]
    greater: list[int] = [x for x in arr if x > pivot]
    return quicksort(less) + equal + quicksort(greater)
''',
        )
    )

    snippets.append(
        Snippet(
            name="counting_sort",
            category="Classic Algorithms and Data Structures",
            code='''def counting_sort(arr: list[int]) -> list[int]:
    """Sort non-negative integers in linear time via frequency counting.

    Args:
        arr: List of non-negative integers.

    Returns:
        Sorted integer list.

    Raises:
        ValueError: If arr contains negative values.
    """
    if not arr:
        return []
    if any(x < 0 for x in arr):
        raise ValueError("Counting sort expects non-negative integers.")
    max_val: int = max(arr)
    counts: list[int] = [0] * (max_val + 1)
    for num in arr:
        counts[num] += 1
    result: list[int] = []
    for val, count in enumerate(counts):
        result.extend([val] * count)
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="Stack",
            category="Classic Algorithms and Data Structures",
            code='''class Stack:
    """A Last In First Out (LIFO) stack data structure."""

    def __init__(self) -> None:
        """Initialize an empty stack collection."""
        self._items: list[int] = []

    def push(self, item: int) -> None:
        """Push a value onto the top of the stack.

        Args:
            item: Integer element to add.
        """
        self._items.append(item)

    def pop(self) -> int:
        """Remove and return the topmost item from the stack.

        Returns:
            The popped integer value.

        Raises:
            IndexError: If the stack has no items.
        """
        if self.is_empty():
            raise IndexError("Cannot pop from an empty stack.")
        return self._items.pop()

    def peek(self) -> int:
        """Inspect the top element without removing it.

        Returns:
            Current topmost integer.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot peek into an empty stack.")
        return self._items[-1]

    def is_empty(self) -> bool:
        """Check whether the stack contains zero elements.

        Returns:
            True if empty, False otherwise.
        """
        return len(self._items) == 0

    def size(self) -> int:
        """Get the count of items stored in the stack.

        Returns:
            Number of elements currently in the stack.
        """
        return len(self._items)
''',
        )
    )

    snippets.append(
        Snippet(
            name="Queue",
            category="Classic Algorithms and Data Structures",
            code='''class Queue:
    """A First In First Out (FIFO) queue data structure."""

    def __init__(self) -> None:
        """Initialize an empty queue."""
        self._elements: list[int] = []

    def enqueue(self, item: int) -> None:
        """Append an element to the back of the queue.

        Args:
            item: Integer item to enqueue.
        """
        self._elements.append(item)

    def dequeue(self) -> int:
        """Extract the front element from the queue.

        Returns:
            The front integer item.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot dequeue from an empty queue.")
        return self._elements.pop(0)

    def peek(self) -> int:
        """Return the next element in line without removing it.

        Returns:
            Front integer value.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot peek into an empty queue.")
        return self._elements[0]

    def is_empty(self) -> bool:
        """Check whether the queue is devoid of elements.

        Returns:
            True if empty, False otherwise.
        """
        return len(self._elements) == 0

    def size(self) -> int:
        """Report total number of queued elements.

        Returns:
            Current queue size.
        """
        return len(self._elements)
''',
        )
    )

    snippets.append(
        Snippet(
            name="Node",
            category="Classic Algorithms and Data Structures",
            code='''class Node:
    """Singly-linked list node holding an integer payload."""

    def __init__(self, value: int, next_node: Optional[Node] = None) -> None:
        """Initialize node with payload and reference to successor.

        Args:
            value: Integer data payload.
            next_node: Optional successor node reference.
        """
        self.value: int = value
        self.next: Optional[Node] = next_node

    def __repr__(self) -> str:
        """Produce readable representation of node.

        Returns:
            String description of the node.
        """
        return f"Node({self.value})"
''',
        )
    )

    snippets.append(
        Snippet(
            name="LinkedList",
            category="Classic Algorithms and Data Structures",
            code='''class LinkedList:
    """Singly linked list container providing fundamental operations."""

    def __init__(self) -> None:
        """Initialize an empty singly linked list."""
        self.head: Optional[Node] = None
        self._length: int = 0

    def append(self, value: int) -> None:
        """Attach a new integer to the end of the list.

        Args:
            value: Integer value to append.
        """
        new_node: Node = Node(value)
        if self.head is None:
            self.head = new_node
        else:
            current: Node = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node
        self._length += 1

    def prepend(self, value: int) -> None:
        """Insert a new integer at the head of the list.

        Args:
            value: Integer value to prepend.
        """
        self.head = Node(value, next_node=self.head)
        self._length += 1

    def delete(self, value: int) -> bool:
        """Remove the first occurrence of a value from the list.

        Args:
            value: Target value to excise.

        Returns:
            True if an item was found and deleted, False otherwise.
        """
        if self.head is None:
            return False
        if self.head.value == value:
            self.head = self.head.next
            self._length -= 1
            return True
        current: Node = self.head
        while current.next is not None:
            if current.next.value == value:
                current.next = current.next.next
                self._length -= 1
                return True
            current = current.next
        return False

    def to_list(self) -> list[int]:
        """Convert all node values to a standard Python list.

        Returns:
            List of integers in linked traversal order.
        """
        result: list[int] = []
        current: Optional[Node] = self.head
        while current is not None:
            result.append(current.value)
            current = current.next
        return result

    def size(self) -> int:
        """Return total count of nodes in the linked list.

        Returns:
            Integer size of the list.
        """
        return self._length
''',
        )
    )

    snippets.append(
        Snippet(
            name="TreeNode",
            category="Classic Algorithms and Data Structures",
            code='''class TreeNode:
    """Binary search tree node holding a numerical key."""

    def __init__(self, key: int) -> None:
        """Initialize a tree node with key and null child pointers.

        Args:
            key: Integer search key.
        """
        self.key: int = key
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None
''',
        )
    )

    snippets.append(
        Snippet(
            name="BinarySearchTree",
            category="Classic Algorithms and Data Structures",
            code='''class BinarySearchTree:
    """Binary Search Tree providing insertion and search operations."""

    def __init__(self) -> None:
        """Initialize an empty binary search tree."""
        self.root: Optional[TreeNode] = None

    def insert(self, key: int) -> None:
        """Insert a key into the binary search tree.

        Args:
            key: Integer key to insert.
        """
        if self.root is None:
            self.root = TreeNode(key)
            return

        current: TreeNode = self.root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = TreeNode(key)
                    break
                current = current.left
            elif key > current.key:
                if current.right is None:
                    current.right = TreeNode(key)
                    break
                current = current.right
            else:
                break

    def search(self, key: int) -> bool:
        """Determine whether a given key exists in the tree.

        Args:
            key: Target search key.

        Returns:
            True if key is present, False otherwise.
        """
        current: Optional[TreeNode] = self.root
        while current is not None:
            if key == current.key:
                return True
            if key < current.key:
                current = current.left
            else:
                current = current.right
        return False

    def inorder_traversal(self) -> list[int]:
        """Produce sorted list of keys using in-order traversal.

        Returns:
            Ordered list of tree keys.
        """
        result: list[int] = []

        def _traverse(node: Optional[TreeNode]) -> None:
            if node is not None:
                _traverse(node.left)
                result.append(node.key)
                _traverse(node.right)

        _traverse(self.root)
        return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="PriorityQueue",
            category="Classic Algorithms and Data Structures",
            code='''class PriorityQueue:
    """A min-priority queue based on ordered list maintenance."""

    def __init__(self) -> None:
        """Initialize an empty priority queue."""
        self._entries: list[tuple[float, str]] = []

    def push(self, priority: float, item: str) -> None:
        """Insert an item with its associated numerical priority.

        Args:
            priority: Numerical rank where smaller indicates higher priority.
            item: String payload.
        """
        self._entries.append((priority, item))
        self._entries.sort(key=lambda pair: pair[0])

    def pop(self) -> str:
        """Remove and return the payload with highest priority.

        Returns:
            String item with smallest priority number.

        Raises:
            IndexError: If priority queue has no items.
        """
        if self.is_empty():
            raise IndexError("Cannot pop from an empty priority queue.")
        return self._entries.pop(0)[1]

    def peek(self) -> str:
        """Inspect the highest priority payload without removal.

        Returns:
            String item with smallest priority score.

        Raises:
            IndexError: If priority queue is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot peek into empty priority queue.")
        return self._entries[0][1]

    def is_empty(self) -> bool:
        """Check whether the priority queue has no entries.

        Returns:
            True if empty, False otherwise.
        """
        return len(self._entries) == 0

    def size(self) -> int:
        """Get the count of items waiting in queue.

        Returns:
            Number of entries.
        """
        return len(self._entries)
''',
        )
    )

    snippets.append(
        Snippet(
            name="breadth_first_search",
            category="Classic Algorithms and Data Structures",
            code='''def breadth_first_search(graph: dict[str, list[str]], start: str) -> list[str]:
    """Traverse an adjacency graph in breadth-first layer order.

    Args:
        graph: Mapping of node labels to list of neighbor nodes.
        start: Label of initial root node.

    Returns:
        List of visited nodes in discovery sequence.
    """
    if start not in graph:
        return []
    visited: set[str] = {start}
    queue: list[str] = [start]
    traversal: list[str] = []
    while queue:
        node: str = queue.pop(0)
        traversal.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return traversal
''',
        )
    )

    snippets.append(
        Snippet(
            name="depth_first_search",
            category="Classic Algorithms and Data Structures",
            code='''def depth_first_search(graph: dict[str, list[str]], start: str) -> list[str]:
    """Traverse a graph depth-first starting at a chosen source.

    Args:
        graph: Adjacency list mapping node labels to neighbors.
        start: Label of the origin node.

    Returns:
        Sequence of nodes visited.
    """
    if start not in graph:
        return []
    visited: set[str] = set()
    order: list[str] = []

    def _dfs(node: str) -> None:
        visited.add(node)
        order.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                _dfs(neighbor)

    _dfs(start)
    return order
''',
        )
    )

    snippets.append(
        Snippet(
            name="dijkstra_shortest_path",
            category="Classic Algorithms and Data Structures",
            code='''def dijkstra_shortest_path(
    graph: dict[str, dict[str, float]], start: str
) -> dict[str, float]:
    """Compute shortest distances from start vertex to all reachable nodes.

    Args:
        graph: Nested mapping from node to neighbors and edge weights.
        start: Origin node name.

    Returns:
        Dictionary mapping node names to minimal distance values.
    """
    distances: dict[str, float] = {node: float("inf") for node in graph}
    distances[start] = 0.0
    unvisited: set[str] = set(graph.keys())

    while unvisited:
        current: Optional[str] = None
        current_dist: float = float("inf")
        for node in unvisited:
            if distances[node] < current_dist:
                current_dist = distances[node]
                current = node

        if current is None or current_dist == float("inf"):
            break

        unvisited.remove(current)
        for neighbor, weight in graph[current].items():
            candidate: float = current_dist + weight
            if candidate < distances.get(neighbor, float("inf")):
                distances[neighbor] = candidate

    return distances
''',
        )
    )

    snippets.append(
        Snippet(
            name="has_cycle_directed",
            category="Classic Algorithms and Data Structures",
            code='''def has_cycle_directed(graph: dict[str, list[str]]) -> bool:
    """Detect whether a directed graph contains any directed cycle.

    Args:
        graph: Adjacency list of directed edges.

    Returns:
        True if at least one cycle exists, False otherwise.
    """
    visited: set[str] = set()
    recursion_stack: set[str] = set()

    def _check(node: str) -> bool:
        visited.add(node)
        recursion_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if _check(neighbor):
                    return True
            elif neighbor in recursion_stack:
                return True
        recursion_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            if _check(node):
                return True
    return False
''',
        )
    )

    snippets.append(
        Snippet(
            name="topological_sort",
            category="Classic Algorithms and Data Structures",
            code='''def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Perform topological ordering on a directed acyclic graph.

    Args:
        graph: Mapping from each vertex to its outgoing targets.

    Returns:
        List of vertices ordered such that dependencies appear first.

    Raises:
        ValueError: If a directed cycle is present.
    """
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for targets in graph.values():
        for target in targets:
            in_degree[target] = in_degree.get(target, 0) + 1

    queue: list[str] = [node for node, deg in in_degree.items() if deg == 0]
    result: list[str] = []

    while queue:
        curr: str = queue.pop(0)
        result.append(curr)
        for neighbor in graph.get(curr, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(in_degree):
        raise ValueError("Graph contains a cycle; topological sort impossible.")
    return result
''',
        )
    )

    # =========================================================================
    # Category 3: TinyTorch and Deep Learning Primitives
    # =========================================================================

    snippets.append(
        Snippet(
            name="relu",
            category="TinyTorch & Deep Learning Primitives",
            code='''def relu(x: float) -> float:
    """Compute Rectified Linear Unit activation on a scalar.

    Args:
        x: Input scalar value.

    Returns:
        Activated output zero or positive.
    """
    return x if x > 0.0 else 0.0
''',
        )
    )

    snippets.append(
        Snippet(
            name="relu_backward",
            category="TinyTorch & Deep Learning Primitives",
            code='''def relu_backward(grad_output: list[float], x: list[float]) -> list[float]:
    """Compute gradients flowing backward through a ReLU activation layer.

    Args:
        grad_output: Upstream gradient vector.
        x: Pre-activation forward input vector.

    Returns:
        Downstream gradient vector with respect to input.
    """
    return [g if val > 0.0 else 0.0 for g, val in zip(grad_output, x)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="sigmoid",
            category="TinyTorch & Deep Learning Primitives",
            code='''def sigmoid(x: float) -> float:
    """Compute logistic sigmoid function bounded in interval (0, 1).

    Args:
        x: Input scalar value.

    Returns:
        Sigmoidal probability response.
    """
    if x >= 0.0:
        z: float = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
''',
        )
    )

    snippets.append(
        Snippet(
            name="sigmoid_backward",
            category="TinyTorch & Deep Learning Primitives",
            code='''def sigmoid_backward(
    grad_output: list[float], activated: list[float]
) -> list[float]:
    """Compute derivative of sigmoid activation given forward outputs.

    Args:
        grad_output: Upstream gradient list.
        activated: Pre-computed sigmoid activations.

    Returns:
        Gradient list propagated backward.
    """
    return [g * a * (1.0 - a) for g, a in zip(grad_output, activated)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="tanh",
            category="TinyTorch & Deep Learning Primitives",
            code='''def tanh(x: float) -> float:
    """Calculate hyperbolic tangent activation bounded in (-1, 1).

    Args:
        x: Real-valued scalar input.

    Returns:
        Hyperbolic tangent value.
    """
    pos_exp: float = math.exp(x)
    neg_exp: float = math.exp(-x)
    return (pos_exp - neg_exp) / (pos_exp + neg_exp)
''',
        )
    )

    snippets.append(
        Snippet(
            name="tanh_backward",
            category="TinyTorch & Deep Learning Primitives",
            code='''def tanh_backward(
    grad_output: list[float], activated: list[float]
) -> list[float]:
    """Compute gradient for tanh given activated values.

    Args:
        grad_output: Incoming gradient sequence.
        activated: Pre-calculated tanh activations.

    Returns:
        Downstream gradients.
    """
    return [g * (1.0 - a * a) for g, a in zip(grad_output, activated)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="gelu",
            category="TinyTorch & Deep Learning Primitives",
            code='''def gelu(x: float) -> float:
    """Compute Gaussian Error Linear Unit activation with tanh approximation.

    Args:
        x: Input scalar coordinate.

    Returns:
        GELU activated output.
    """
    inner: float = math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))
    cdf_approx: float = 0.5 * (1.0 + math.tanh(inner))
    return x * cdf_approx
''',
        )
    )

    snippets.append(
        Snippet(
            name="leaky_relu",
            category="TinyTorch & Deep Learning Primitives",
            code='''def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Compute Leaky ReLU activation with specified negative slope.

    Args:
        x: Scalar input argument.
        alpha: Small positive scaling factor for negative values.

    Returns:
        Leaky activation result.
    """
    return x if x > 0.0 else alpha * x
''',
        )
    )

    snippets.append(
        Snippet(
            name="softmax",
            category="TinyTorch & Deep Learning Primitives",
            code='''def softmax(logits: list[float]) -> list[float]:
    """Compute numerically stable softmax distribution over logits.

    Args:
        logits: Unnormalized class score vector.

    Returns:
        Normalized categorical probability distribution.

    Raises:
        ValueError: If logits vector is empty.
    """
    if not logits:
        raise ValueError("Cannot apply softmax to empty list.")
    max_logit: float = max(logits)
    shifted_exps: list[float] = [math.exp(val - max_logit) for val in logits]
    sum_exps: float = sum(shifted_exps)
    return [val / sum_exps for val in shifted_exps]
''',
        )
    )

    snippets.append(
        Snippet(
            name="log_softmax",
            category="TinyTorch & Deep Learning Primitives",
            code='''def log_softmax(logits: list[float]) -> list[float]:
    """Compute numerically stable logarithm of softmax distribution.

    Args:
        logits: Raw unnormalized prediction values.

    Returns:
        Vector of log-probability values.

    Raises:
        ValueError: If logits is empty.
    """
    if not logits:
        raise ValueError("Logits list cannot be empty.")
    max_logit: float = max(logits)
    sum_exps: float = sum(math.exp(val - max_logit) for val in logits)
    log_sum_exp: float = max_logit + math.log(sum_exps)
    return [val - log_sum_exp for val in logits]
''',
        )
    )

    snippets.append(
        Snippet(
            name="mse_loss",
            category="TinyTorch & Deep Learning Primitives",
            code='''def mse_loss(predictions: list[float], targets: list[float]) -> float:
    """Compute mean squared error loss between predictions and targets.

    Args:
        predictions: Model output predictions.
        targets: True ground truth values.

    Returns:
        Mean squared error scalar.

    Raises:
        ValueError: If dimensions mismatch or lists are empty.
    """
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Lengths must be identical and non-zero.")
    total_sq_error: float = sum(
        (p - t) ** 2 for p, t in zip(predictions, targets)
    )
    return total_sq_error / len(predictions)
''',
        )
    )

    snippets.append(
        Snippet(
            name="mse_loss_backward",
            category="TinyTorch & Deep Learning Primitives",
            code='''def mse_loss_backward(
    predictions: list[float], targets: list[float]
) -> list[float]:
    """Compute gradient of mean squared error loss with respect to predictions.

    Args:
        predictions: Model predictions.
        targets: Target ground truth values.

    Returns:
        Gradient vector for predictions.
    """
    n: int = len(predictions)
    scale: float = 2.0 / n
    return [scale * (p - t) for p, t in zip(predictions, targets)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="cross_entropy_loss",
            category="TinyTorch & Deep Learning Primitives",
            code='''def cross_entropy_loss(logits: list[float], target_idx: int) -> float:
    """Compute multiclass cross entropy loss given unnormalized logits.

    Args:
        logits: Raw predictions for each class.
        target_idx: Ground truth class index.

    Returns:
        Negative log likelihood scalar.

    Raises:
        IndexError: If target index is out of bounds.
    """
    if target_idx < 0 or target_idx >= len(logits):
        raise IndexError("Target class index is outside logits bounds.")
    max_logit: float = max(logits)
    sum_exps: float = sum(math.exp(val - max_logit) for val in logits)
    log_sum_exp: float = max_logit + math.log(sum_exps)
    return -(logits[target_idx] - log_sum_exp)
''',
        )
    )

    snippets.append(
        Snippet(
            name="bce_loss",
            category="TinyTorch & Deep Learning Primitives",
            code='''def bce_loss(
    predictions: list[float], targets: list[float], eps: float = 1e-15
) -> float:
    """Compute binary cross entropy loss for probability predictions.

    Args:
        predictions: Predicted probabilities in range (0, 1).
        targets: Binary labels either 0.0 or 1.0.
        eps: Numerical stability epsilon to avoid log of zero.

    Returns:
        Binary cross entropy scalar loss.

    Raises:
        ValueError: If lengths do not match or are empty.
    """
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("Inputs must have identical non-zero lengths.")
    total_loss: float = 0.0
    for p, y in zip(predictions, targets):
        clamped_p: float = max(eps, min(1.0 - eps, p))
        total_loss += -(y * math.log(clamped_p) + (1.0 - y) * math.log(1.0 - clamped_p))
    return total_loss / len(predictions)
''',
        )
    )

    snippets.append(
        Snippet(
            name="linear_forward",
            category="TinyTorch & Deep Learning Primitives",
            code='''def linear_forward(
    x: list[float], weights: list[list[float]], bias: list[float]
) -> list[float]:
    """Perform forward matrix-vector affine transformation y = Wx + b.

    Args:
        x: Input feature vector of size in_features.
        weights: Weight matrix of shape [out_features, in_features].
        bias: Bias offset vector of size out_features.

    Returns:
        Output feature vector of size out_features.

    Raises:
        ValueError: If inner dimensions mismatch.
    """
    out_dim: int = len(weights)
    in_dim: int = len(x)
    output: list[float] = [0.0] * out_dim
    for i in range(out_dim):
        if len(weights[i]) != in_dim:
            raise ValueError("Weight row length does not match input length.")
        dot: float = sum(weights[i][j] * x[j] for j in range(in_dim))
        output[i] = dot + bias[i]
    return output
''',
        )
    )

    snippets.append(
        Snippet(
            name="linear_backward",
            category="TinyTorch & Deep Learning Primitives",
            code='''def linear_backward(
    x: list[float],
    grad_output: list[float],
    weights: list[list[float]],
) -> tuple[list[float], list[list[float]], list[float]]:
    """Compute gradients of affine layer with respect to input, weights, bias.

    Args:
        x: Input feature activations of size in_features.
        grad_output: Upstream gradient of size out_features.
        weights: Weight matrix of shape [out_features, in_features].

    Returns:
        Tuple containing grad_input, grad_weights, and grad_bias.
    """
    out_dim: int = len(weights)
    in_dim: int = len(x)

    grad_input: list[float] = [0.0] * in_dim
    for j in range(in_dim):
        grad_input[j] = sum(grad_output[i] * weights[i][j] for i in range(out_dim))

    grad_weights: list[list[float]] = [
        [grad_output[i] * x[j] for j in range(in_dim)] for i in range(out_dim)
    ]
    grad_bias: list[float] = list(grad_output)

    return grad_input, grad_weights, grad_bias
''',
        )
    )

    snippets.append(
        Snippet(
            name="dropout_forward",
            category="TinyTorch & Deep Learning Primitives",
            code='''def dropout_forward(
    x: list[float], mask: list[float], p: float, training: bool
) -> list[float]:
    """Apply inverted dropout forward pass given an explicit binary mask.

    Args:
        x: Input feature vector.
        mask: Binary mask with 1.0 for keep and 0.0 for drop.
        p: Dropout probability indicating fraction of units dropped.
        training: Boolean flag signaling training phase.

    Returns:
        Scaled output vector with dropped activations zeroed.
    """
    if not training or p == 0.0:
        return list(x)
    scale: float = 1.0 / (1.0 - p)
    return [val * m * scale for val, m in zip(x, mask)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="clip_grad_norm",
            category="TinyTorch & Deep Learning Primitives",
            code='''def clip_grad_norm(gradients: list[float], max_norm: float) -> list[float]:
    """Scale down gradient vector if its Euclidean norm exceeds a threshold.

    Args:
        gradients: Flattened list of gradient components.
        max_norm: Maximum permitted global Euclidean norm.

    Returns:
        Clipped gradients respecting the threshold constraint.
    """
    total_norm: float = math.sqrt(sum(g * g for g in gradients))
    if total_norm <= max_norm or total_norm == 0.0:
        return list(gradients)
    scale: float = max_norm / total_norm
    return [g * scale for g in gradients]
''',
        )
    )

    snippets.append(
        Snippet(
            name="layer_norm",
            category="TinyTorch & Deep Learning Primitives",
            code='''def layer_norm(
    x: list[float], gamma: list[float], beta: list[float], eps: float = 1e-5
) -> list[float]:
    """Standardize vector activations across features with affine modulation.

    Args:
        x: Input activation vector.
        gamma: Learnable gain scaling vector.
        beta: Learnable bias shift vector.
        eps: Small stability constant to prevent division by zero.

    Returns:
        Normalized and modulated vector.
    """
    n: int = len(x)
    mu: float = sum(x) / n
    var: float = sum((val - mu) ** 2 for val in x) / n
    std: float = math.sqrt(var + eps)
    return [((val - mu) / std) * g + b for val, g, b in zip(x, gamma, beta)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="batch_norm1d_inference",
            category="TinyTorch & Deep Learning Primitives",
            code='''def batch_norm1d_inference(
    x: list[float],
    running_mean: list[float],
    running_var: list[float],
    gamma: list[float],
    beta: list[float],
    eps: float = 1e-5,
) -> list[float]:
    """Normalize features during evaluation using tracked population statistics.

    Args:
        x: Feature vector for a single sample.
        running_mean: Running average of feature means.
        running_var: Running average of feature variances.
        gamma: Affine weight scaling parameter.
        beta: Affine bias offset parameter.
        eps: Numerical stability constant.

    Returns:
        Normalized and transformed output vector.
    """
    output: list[float] = []
    for val, mean_val, var_val, g, b in zip(
        x, running_mean, running_var, gamma, beta
    ):
        normed: float = (val - mean_val) / math.sqrt(var_val + eps)
        output.append(normed * g + b)
    return output
''',
        )
    )

    snippets.append(
        Snippet(
            name="one_hot_encode",
            category="TinyTorch & Deep Learning Primitives",
            code='''def one_hot_encode(indices: list[int], num_classes: int) -> list[list[float]]:
    """Convert integer class label indices into one-hot indicator vectors.

    Args:
        indices: List of integer class targets.
        num_classes: Total count of categories.

    Returns:
        List of binary one-hot vectors.
    """
    result: list[list[float]] = []
    for idx in indices:
        vec: list[float] = [0.0] * num_classes
        if 0 <= idx < num_classes:
            vec[idx] = 1.0
        result.append(vec)
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="Tensor",
            category="TinyTorch & Deep Learning Primitives",
            code='''class Tensor:
    """A minimal 1D tensor with reverse-mode automatic differentiation."""

    def __init__(self, data: list[float]) -> None:
        """Initialize a tensor with data and zero gradients.

        Args:
            data: List of floating point values.
        """
        self.data: list[float] = list(data)
        self.grad: list[float] = [0.0] * len(data)
        self._backward: Optional[callable] = None
        self._prev: set[Tensor] = set()

    def zero_grad(self) -> None:
        """Reset accumulated gradients to zero."""
        self.grad = [0.0] * len(self.data)

    def add(self, other: Tensor) -> Tensor:
        """Element-wise addition of two tensors.

        Args:
            other: Second tensor operand.

        Returns:
            Resulting tensor after addition.
        """
        out = Tensor([a + b for a, b in zip(self.data, other.data)])
        out._prev = {self, other}

        def _backward() -> None:
            for i in range(len(self.data)):
                self.grad[i] += out.grad[i]
                other.grad[i] += out.grad[i]

        out._backward = _backward
        return out

    def mul(self, other: Tensor) -> Tensor:
        """Element-wise multiplication of two tensors.

        Args:
            other: Multiplier tensor.

        Returns:
            New tensor containing product.
        """
        out = Tensor([a * b for a, b in zip(self.data, other.data)])
        out._prev = {self, other}

        def _backward() -> None:
            for i in range(len(self.data)):
                self.grad[i] += other.data[i] * out.grad[i]
                other.grad[i] += self.data[i] * out.grad[i]

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        """Apply element-wise ReLU activation.

        Returns:
            New activated tensor.
        """
        out = Tensor([val if val > 0.0 else 0.0 for val in self.data])
        out._prev = {self}

        def _backward() -> None:
            for i in range(len(self.data)):
                factor: float = 1.0 if self.data[i] > 0.0 else 0.0
                self.grad[i] += factor * out.grad[i]

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Execute backpropagation through the computational graph."""
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def _build_topo(node: Tensor) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    _build_topo(child)
                topo.append(node)

        _build_topo(self)
        self.grad = [1.0] * len(self.data)
        for node in reversed(topo):
            if node._backward is not None:
                node._backward()
''',
        )
    )

    snippets.append(
        Snippet(
            name="Linear",
            category="TinyTorch & Deep Learning Primitives",
            code='''class Linear:
    """A fully connected linear neural network layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        """Initialize weights and biases with simple deterministic values.

        Args:
            in_features: Number of incoming inputs.
            out_features: Number of outgoing features.
        """
        self.in_features: int = in_features
        self.out_features: int = out_features
        scale: float = 1.0 / math.sqrt(in_features)
        self.weights: list[list[float]] = [
            [scale * 0.5 for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.bias: list[float] = [0.0] * out_features
        self.grad_weights: list[list[float]] = [
            [0.0 for _ in range(in_features)]
            for _ in range(out_features)
        ]
        self.grad_bias: list[float] = [0.0] * out_features
        self.last_input: list[float] = []

    def forward(self, x: list[float]) -> list[float]:
        """Compute layer output activations.

        Args:
            x: Input vector of size in_features.

        Returns:
            Transformed vector of size out_features.
        """
        self.last_input = list(x)
        output: list[float] = []
        for i in range(self.out_features):
            val: float = sum(
                self.weights[i][j] * x[j] for j in range(self.in_features)
            )
            output.append(val + self.bias[i])
        return output

    def zero_grad(self) -> None:
        """Reset parameter gradients to zero."""
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.grad_weights[i][j] = 0.0
            self.grad_bias[i] = 0.0

    def step(self, lr: float) -> None:
        """Update parameters using accumulated gradients.

        Args:
            lr: Learning rate step factor.
        """
        for i in range(self.out_features):
            for j in range(self.in_features):
                self.weights[i][j] -= lr * self.grad_weights[i][j]
            self.bias[i] -= lr * self.grad_bias[i]
''',
        )
    )

    snippets.append(
        Snippet(
            name="Sequential",
            category="TinyTorch & Deep Learning Primitives",
            code='''class Sequential:
    """A feedforward sequential container chaining layers together."""

    def __init__(self) -> None:
        """Initialize an empty sequential model."""
        self.layers: list[Linear] = []

    def add(self, layer: Linear) -> None:
        """Append a layer to the execution pipeline.

        Args:
            layer: Linear module to append.
        """
        self.layers.append(layer)

    def forward(self, x: list[float]) -> list[float]:
        """Propagate input sequentially through all layers.

        Args:
            x: Initial input activations.

        Returns:
            Final output vector from the last layer.
        """
        current: list[float] = list(x)
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def zero_grad(self) -> None:
        """Zero the gradients across all constituent layers."""
        for layer in self.layers:
            layer.zero_grad()

    def step(self, lr: float) -> None:
        """Advance parameter weights across all layers.

        Args:
            lr: Step size for gradient descent.
        """
        for layer in self.layers:
            layer.step(lr)
''',
        )
    )

    snippets.append(
        Snippet(
            name="SGD",
            category="TinyTorch & Deep Learning Primitives",
            code='''class SGD:
    """Stochastic Gradient Descent optimizer with momentum support."""

    def __init__(
        self,
        params: list[list[float]],
        grads: list[list[float]],
        lr: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        """Initialize optimizer hyperparameters and velocity buffers.

        Args:
            params: List of parameter vectors to optimize.
            grads: Matching list of gradient vectors.
            lr: Learning rate scaling factor.
            momentum: Momentum velocity decay rate.
        """
        self.params: list[list[float]] = params
        self.grads: list[list[float]] = grads
        self.lr: float = lr
        self.momentum: float = momentum
        self.velocities: list[list[float]] = [
            [0.0] * len(p) for p in params
        ]

    def step(self) -> None:
        """Perform a single optimization step updating parameters."""
        for p, g, v in zip(self.params, self.grads, self.velocities):
            for i in range(len(p)):
                v[i] = self.momentum * v[i] + g[i]
                p[i] -= self.lr * v[i]

    def zero_grad(self) -> None:
        """Reset parameter gradient buffers to zero."""
        for g in self.grads:
            for i in range(len(g)):
                g[i] = 0.0
''',
        )
    )

    snippets.append(
        Snippet(
            name="Adam",
            category="TinyTorch & Deep Learning Primitives",
            code='''class Adam:
    """Adaptive Moment Estimation optimizer tracking first and second moments."""

    def __init__(
        self,
        params: list[list[float]],
        grads: list[list[float]],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize Adam moments and constants.

        Args:
            params: Parameter lists to optimize.
            grads: Matching gradient vectors.
            lr: Step learning rate.
            beta1: Exponential decay rate for first moments.
            beta2: Exponential decay rate for second moments.
            eps: Epsilon factor to prevent division by zero.
        """
        self.params: list[list[float]] = params
        self.grads: list[list[float]] = grads
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.eps: float = eps
        self.m: list[list[float]] = [[0.0] * len(p) for p in params]
        self.v: list[list[float]] = [[0.0] * len(p) for p in params]
        self.t: int = 0

    def step(self) -> None:
        """Advance parameters using bias-corrected moment estimates."""
        self.t += 1
        correction1: float = 1.0 - (self.beta1 ** self.t)
        correction2: float = 1.0 - (self.beta2 ** self.t)

        for p, g, m_buf, v_buf in zip(
            self.params, self.grads, self.m, self.v
        ):
            for i in range(len(p)):
                grad_val: float = g[i]
                m_buf[i] = self.beta1 * m_buf[i] + (1.0 - self.beta1) * grad_val
                v_buf[i] = self.beta2 * v_buf[i] + (1.0 - self.beta2) * (grad_val ** 2)
                m_hat: float = m_buf[i] / correction1
                v_hat: float = v_buf[i] / correction2
                p[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
''',
        )
    )

    # =========================================================================
    # Category 4: Utility and String / List operations
    # =========================================================================

    snippets.append(
        Snippet(
            name="reverse_list",
            category="Utility and String / List operations",
            code='''def reverse_list(items: list[int]) -> list[int]:
    """Invert the sequence ordering of items in a list.

    Args:
        items: List of integers to reverse.

    Returns:
        New list containing elements in reversed sequence.
    """
    result: list[int] = []
    for i in range(len(items) - 1, -1, -1):
        result.append(items[i])
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="flatten",
            category="Utility and String / List operations",
            code='''def flatten(nested: list[list[int]]) -> list[int]:
    """Flatten a two-dimensional nested list into a single list.

    Args:
        nested: Matrix or list of integer sublists.

    Returns:
        One-dimensional list of flattened integers.
    """
    result: list[int] = []
    for sublist in nested:
        for item in sublist:
            result.append(item)
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="chunk_list",
            category="Utility and String / List operations",
            code='''def chunk_list(items: list[int], chunk_size: int) -> list[list[int]]:
    """Partition a list into consecutive sublists of fixed size.

    Args:
        items: List of integers to partition.
        chunk_size: Maximum size of each partition chunk.

    Returns:
        List containing partitioned chunks.

    Raises:
        ValueError: If chunk_size is not strictly positive.
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    result: list[list[int]] = []
    for i in range(0, len(items), chunk_size):
        result.append(items[i : i + chunk_size])
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="filter_positive",
            category="Utility and String / List operations",
            code='''def filter_positive(numbers: list[float]) -> list[float]:
    """Filter a list to retain only strictly positive numbers.

    Args:
        numbers: Sequence of floating point values.

    Returns:
        Filtered list holding numbers greater than zero.
    """
    return [x for x in numbers if x > 0.0]
''',
        )
    )

    snippets.append(
        Snippet(
            name="count_occurrences",
            category="Utility and String / List operations",
            code='''def count_occurrences(items: list[str], target: str) -> int:
    """Count how many times a target string appears in a list.

    Args:
        items: List of strings to search.
        target: Specific string to count.

    Returns:
        Total integer count of matches found.
    """
    count: int = 0
    for s in items:
        if s == target:
            count += 1
    return count
''',
        )
    )

    snippets.append(
        Snippet(
            name="find_max",
            category="Utility and String / List operations",
            code='''def find_max(numbers: list[float]) -> float:
    """Identify the largest numerical value in a list.

    Args:
        numbers: Non-empty list of floating point values.

    Returns:
        Largest value found.

    Raises:
        ValueError: If the list is empty.
    """
    if not numbers:
        raise ValueError("Cannot find maximum of empty list.")
    highest: float = numbers[0]
    for val in numbers[1:]:
        if val > highest:
            highest = val
    return highest
''',
        )
    )

    snippets.append(
        Snippet(
            name="find_min",
            category="Utility and String / List operations",
            code='''def find_min(numbers: list[float]) -> float:
    """Identify the smallest numerical value in a list.

    Args:
        numbers: Non-empty list of floating point numbers.

    Returns:
        Smallest value found.

    Raises:
        ValueError: If the list is empty.
    """
    if not numbers:
        raise ValueError("Cannot find minimum of empty list.")
    lowest: float = numbers[0]
    for val in numbers[1:]:
        if val < lowest:
            lowest = val
    return lowest
''',
        )
    )

    snippets.append(
        Snippet(
            name="normalize_vector",
            category="Utility and String / List operations",
            code='''def normalize_vector(vec: list[float], eps: float = 1e-12) -> list[float]:
    """Rescale a vector to unit Euclidean norm length.

    Args:
        vec: Vector of floating point numbers.
        eps: Small tolerance to handle near-zero vectors.

    Returns:
        Normalized unit vector.
    """
    norm: float = math.sqrt(sum(x * x for x in vec))
    if norm < eps:
        return [0.0] * len(vec)
    return [x / norm for x in vec]
''',
        )
    )

    snippets.append(
        Snippet(
            name="dot_product",
            category="Utility and String / List operations",
            code='''def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute standard dot product between two numerical vectors.

    Args:
        vec_a: First vector of floating point numbers.
        vec_b: Second vector of floating point numbers.

    Returns:
        Inner dot product scalar.

    Raises:
        ValueError: If vector dimensions differ.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have matching lengths.")
    return sum(a * b for a, b in zip(vec_a, vec_b))
''',
        )
    )

    snippets.append(
        Snippet(
            name="euclidean_distance",
            category="Utility and String / List operations",
            code='''def euclidean_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute straight-line Euclidean distance between two points.

    Args:
        vec_a: Coordinates of first vector point.
        vec_b: Coordinates of second vector point.

    Returns:
        Euclidean distance scalar.

    Raises:
        ValueError: If vector dimensions do not match.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimensionalities must match.")
    sum_sq: float = sum((a - b) ** 2 for a, b in zip(vec_a, vec_b))
    return math.sqrt(sum_sq)
''',
        )
    )

    snippets.append(
        Snippet(
            name="cosine_similarity",
            category="Utility and String / List operations",
            code='''def cosine_similarity(
    vec_a: list[float], vec_b: list[float], eps: float = 1e-12
) -> float:
    """Compute cosine of the angle between two multi-dimensional vectors.

    Args:
        vec_a: First feature vector.
        vec_b: Second feature vector.
        eps: Numerical guard to prevent division by zero.

    Returns:
        Cosine similarity bounded in interval [-1, 1].

    Raises:
        ValueError: If vector lengths differ.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must share identical length.")
    dot: float = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a: float = math.sqrt(sum(a * a for a in vec_a))
    norm_b: float = math.sqrt(sum(b * b for b in vec_b))
    denom: float = norm_a * norm_b
    if denom < eps:
        return 0.0
    return dot / denom
''',
        )
    )

    snippets.append(
        Snippet(
            name="manhattan_distance",
            category="Utility and String / List operations",
            code='''def manhattan_distance(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute L1 taxicab distance between two vector coordinates.

    Args:
        vec_a: First point coordinates.
        vec_b: Second point coordinates.

    Returns:
        L1 Manhattan distance scalar.

    Raises:
        ValueError: If lengths do not match.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector lengths must be identical.")
    return sum(abs(a - b) for a, b in zip(vec_a, vec_b))
''',
        )
    )

    snippets.append(
        Snippet(
            name="matrix_transpose",
            category="Utility and String / List operations",
            code='''def matrix_transpose(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose a 2D matrix by flipping rows and columns.

    Args:
        matrix: 2D rectangular grid of numbers.

    Returns:
        Transposed 2D matrix grid.
    """
    if not matrix or not matrix[0]:
        return []
    num_rows: int = len(matrix)
    num_cols: int = len(matrix[0])
    transposed: list[list[float]] = [
        [matrix[r][c] for r in range(num_rows)] for c in range(num_cols)
    ]
    return transposed
''',
        )
    )

    snippets.append(
        Snippet(
            name="matrix_multiply",
            category="Utility and String / List operations",
            code='''def matrix_multiply(
    mat_a: list[list[float]], mat_b: list[list[float]]
) -> list[list[float]]:
    """Compute the matrix product of two 2D numerical matrices.

    Args:
        mat_a: Left matrix of dimension [M, K].
        mat_b: Right matrix of dimension [K, N].

    Returns:
        Resulting matrix of dimension [M, N].

    Raises:
        ValueError: If inner dimensions do not match.
    """
    if not mat_a or not mat_b:
        return []
    m: int = len(mat_a)
    k: int = len(mat_a[0])
    k2: int = len(mat_b)
    n: int = len(mat_b[0])
    if k != k2:
        raise ValueError("Matrix inner dimensions must match.")

    result: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            result[i][j] = sum(mat_a[i][p] * mat_b[p][j] for p in range(k))
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="unique_elements",
            category="Utility and String / List operations",
            code='''def unique_elements(items: list[int]) -> list[int]:
    """Extract distinct elements from a list preserving first encounter order.

    Args:
        items: List of integers possibly containing duplicates.

    Returns:
        List of distinct integers in order of appearance.
    """
    seen: set[int] = set()
    result: list[int] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
''',
        )
    )

    snippets.append(
        Snippet(
            name="zip_lists",
            category="Utility and String / List operations",
            code='''def zip_lists(
    list_a: list[int], list_b: list[int]
) -> list[tuple[int, int]]:
    """Pair corresponding elements from two integer lists.

    Args:
        list_a: First integer list.
        list_b: Second integer list.

    Returns:
        List of two-element integer tuples up to the shortest length.
    """
    limit: int = min(len(list_a), len(list_b))
    return [(list_a[i], list_b[i]) for i in range(limit)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="sliding_window",
            category="Utility and String / List operations",
            code='''def sliding_window(
    items: list[int], window_size: int
) -> list[list[int]]:
    """Extract overlapping sequential windows from an integer list.

    Args:
        items: Source list of integers.
        window_size: Width of the sliding window.

    Returns:
        List of sublists representing contiguous windows.

    Raises:
        ValueError: If window_size is non-positive or exceeds list size.
    """
    if window_size <= 0:
        raise ValueError("Window size must be positive.")
    if window_size > len(items):
        return []
    windows: list[list[int]] = []
    for i in range(len(items) - window_size + 1):
        windows.append(items[i : i + window_size])
    return windows
''',
        )
    )

    snippets.append(
        Snippet(
            name="caesar_cipher",
            category="Utility and String / List operations",
            code='''def caesar_cipher(text: str, shift: int) -> str:
    """Encode or decode text using a Caesar rotational cipher.

    Args:
        text: Input string to transform.
        shift: Number of positions to rotate each letter.

    Returns:
        Rotated output string with casing preserved.
    """
    shifted_chars: list[str] = []
    shift = shift % 26
    for char in text:
        if "a" <= char <= "z":
            base: int = ord("a")
            rotated: int = (ord(char) - base + shift) % 26 + base
            shifted_chars.append(chr(rotated))
        elif "A" <= char <= "Z":
            base = ord("A")
            rotated = (ord(char) - base + shift) % 26 + base
            shifted_chars.append(chr(rotated))
        else:
            shifted_chars.append(char)
    return "".join(shifted_chars)
''',
        )
    )

    snippets.append(
        Snippet(
            name="is_palindrome",
            category="Utility and String / List operations",
            code='''def is_palindrome(text: str) -> bool:
    """Check if an alphanumeric sequence reads the same backwards.

    Args:
        text: String to test.

    Returns:
        True if string is palindromic, False otherwise.
    """
    cleaned: list[str] = [c.lower() for c in text if c.isalnum()]
    left: int = 0
    right: int = len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1
    return True
''',
        )
    )

    snippets.append(
        Snippet(
            name="tokenize_whitespace",
            category="Utility and String / List operations",
            code='''def tokenize_whitespace(text: str) -> list[str]:
    """Split text into a sequence of whitespace-delimited tokens.

    Args:
        text: Raw text string.

    Returns:
        List of non-empty token substrings.
    """
    tokens: list[str] = []
    current_token: list[str] = []
    for char in text:
        if char.isspace():
            if current_token:
                tokens.append("".join(current_token))
                current_token = []
        else:
            current_token.append(char)
    if current_token:
        tokens.append("".join(current_token))
    return tokens
''',
        )
    )

    snippets.append(
        Snippet(
            name="char_ngrams",
            category="Utility and String / List operations",
            code='''def char_ngrams(text: str, n: int) -> list[str]:
    """Extract consecutive character n-grams from a text string.

    Args:
        text: Source text sequence.
        n: Window length of each n-gram.

    Returns:
        List of extracted n-gram strings.

    Raises:
        ValueError: If n is not strictly positive.
    """
    if n <= 0:
        raise ValueError("N-gram length must be positive.")
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]
''',
        )
    )

    snippets.append(
        Snippet(
            name="pad_sequence",
            category="Utility and String / List operations",
            code='''def pad_sequence(
    seq: list[int], max_len: int, pad_value: int = 0
) -> list[int]:
    """Pad or truncate an integer sequence to an exact uniform length.

    Args:
        seq: Source integer sequence.
        max_len: Desired target length.
        pad_value: Filler integer to append when length is insufficient.

    Returns:
        Padded or truncated integer list of length max_len.

    Raises:
        ValueError: If max_len is negative.
    """
    if max_len < 0:
        raise ValueError("Target length cannot be negative.")
    if len(seq) >= max_len:
        return seq[:max_len]
    padding_needed: int = max_len - len(seq)
    return list(seq) + [pad_value] * padding_needed
''',
        )
    )

    snippets.append(
        Snippet(
            name="run_length_encode",
            category="Utility and String / List operations",
            code='''def run_length_encode(text: str) -> list[tuple[str, int]]:
    """Compress repeated adjacent characters using run-length encoding.

    Args:
        text: Source string to encode.

    Returns:
        List of (character, count) pairs.
    """
    if not text:
        return []
    encoded: list[tuple[str, int]] = []
    current_char: str = text[0]
    count: int = 1
    for char in text[1:]:
        if char == current_char:
            count += 1
        else:
            encoded.append((current_char, count))
            current_char = char
            count = 1
    encoded.append((current_char, count))
    return encoded
''',
        )
    )

    snippets.append(
        Snippet(
            name="run_length_decode",
            category="Utility and String / List operations",
            code='''def run_length_decode(encoded: list[tuple[str, int]]) -> str:
    """Reconstruct an original string from run-length encoded pairs.

    Args:
        encoded: List of (character, count) tuples.

    Returns:
        Decoded text string.
    """
    pieces: list[str] = [char * count for char, count in encoded]
    return "".join(pieces)
''',
        )
    )

    snippets.append(
        Snippet(
            name="word_frequencies",
            category="Utility and String / List operations",
            code='''def word_frequencies(words: list[str]) -> dict[str, int]:
    """Compute occurrence counts for each word in a list.

    Args:
        words: Collection of word tokens.

    Returns:
        Dictionary mapping distinct words to their frequency counts.
    """
    frequencies: dict[str, int] = {}
    for word in words:
        frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies
''',
        )
    )

    snippets.append(
        Snippet(
            name="levenshtein_distance",
            category="Utility and String / List operations",
            code='''def levenshtein_distance(str_a: str, str_b: str) -> int:
    """Compute the minimum edit distance between two strings.

    Args:
        str_a: Source word string.
        str_b: Target word string.

    Returns:
        Minimum number of insertions, deletions, or substitutions.
    """
    len_a: int = len(str_a)
    len_b: int = len(str_b)
    dp: list[list[int]] = [
        [0 for _ in range(len_b + 1)] for _ in range(len_a + 1)
    ]

    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if str_a[i - 1] == str_b[j - 1]:
                cost: int = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[len_a][len_b]
''',
        )
    )

    return snippets


def build_corpus(snippets: list[Snippet]) -> str:
    """Combine snippets into a formatted plain text corpus file.

    Args:
        snippets: List of Snippet objects.

    Returns:
        Complete formatted corpus string.
    """
    lines: list[str] = []
    lines.append('"""TinyPy Educational Code Corpus for TinyTorch.')
    lines.append("")
    lines.append("A curated collection of clean, idiomatic Python snippets")
    lines.append("spanning arithmetic, classic data structures, deep learning")
    lines.append("primitives, and list/string utilities.")
    lines.append('"""')
    lines.append("")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import math")
    lines.append("from typing import Callable, Optional")
    lines.append("")

    categories_seen: set[str] = set()

    for snippet in snippets:
        if snippet.category not in categories_seen:
            categories_seen.add(snippet.category)
            lines.append("")
            lines.append(f"# {'=' * 75}")
            lines.append(f"# Category: {snippet.category}")
            lines.append(f"# {'=' * 75}")
            lines.append("")

        lines.append(snippet.code.rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def verify_snippets(snippets: list[Snippet]) -> bool:
    """Run ast.parse on every individual snippet and verify syntax.

    Args:
        snippets: List of Snippet instances.

    Returns:
        True if all snippets pass syntax verification, False otherwise.
    """
    total: int = len(snippets)
    category_counts: dict[str, int] = {}
    ast_node_types: set[str] = set()
    errors: list[str] = []

    for snippet in snippets:
        category_counts[snippet.category] = (
            category_counts.get(snippet.category, 0) + 1
        )
        try:
            tree = ast.parse(snippet.code)
            for node in ast.walk(tree):
                ast_node_types.add(type(node).__name__)
        except SyntaxError as e:
            errors.append(f"Snippet '{snippet.name}': {e}")

    print("==================================================")
    print("TinyPy Syntax Verification Report")
    print("==================================================")
    for cat, count in category_counts.items():
        print(f"  * {cat}: {count} snippets")
    print(f"  * Total Snippets: {total}")
    print(f"  * Unique AST Node Types: {len(ast_node_types)}")

    if errors:
        print("Verification FAILED with errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    # Also verify the whole combined corpus
    corpus = build_corpus(snippets)
    try:
        ast.parse(corpus)
        print("  * Full Combined Corpus AST Parse: PASSED")
    except SyntaxError as e:
        print(f"Full Corpus AST Parse FAILED: {e}")
        return False

    byte_count = len(corpus.encode("utf-8"))
    line_count = len(corpus.splitlines())
    print(f"  * Corpus Lines: {line_count}")
    print(f"  * Corpus Size: {byte_count:,} bytes ({byte_count / 1024:.2f} KB)")
    print("  * Overall Syntax Verification: ALL PASS")
    print("==================================================")
    return True


def main() -> None:
    """Main CLI entry point for generating and verifying TinyPy."""
    parser = argparse.ArgumentParser(
        description="Generate and verify the TinyPy code completion dataset."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify syntax of each snippet using ast.parse.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path where output corpus file should be written.",
    )

    args = parser.parse_args()
    snippets = get_snippets()

    if args.verify:
        success = verify_snippets(snippets)
        if not success:
            sys.exit(1)

    if args.output:
        corpus = build_corpus(snippets)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(corpus, encoding="utf-8")
        print(f"TinyPy dataset successfully written to: {out_path.resolve()}")
    elif not args.verify:
        # Default behavior when no flag passed: write default sample
        corpus = build_corpus(snippets)
        default_out = Path(__file__).parent / "tinypy_sample.txt"
        default_out.write_text(corpus, encoding="utf-8")
        print(f"TinyPy dataset written to default path: {default_out.resolve()}")


if __name__ == "__main__":
    main()
