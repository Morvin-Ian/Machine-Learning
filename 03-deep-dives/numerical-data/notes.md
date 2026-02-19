````markdown
# Numerical Data (Practical Notes)

This note covers practical topics related to numerical data that are useful across machine learning workflows: numeric types, precision, stability, and common pitfalls.

## Numeric Types and Precision
- Floating point: `float32` (single) and `float64` (double) are most common. `float64` offers better precision but uses more memory; deep learning often uses `float32` for speed and memory efficiency.
- Integers: `int` types are used for counts/indices. Avoid integer arithmetic when you need averages or divisions.

## Floating-Point Issues
- **Rounding errors:** Floating point numbers are approximations; sums or differences of numbers with very different magnitudes can lose precision. 

  ```python
  # adding a very small number to a large number
  a = 1e16
  b = 1.0
  print(a + b - a)  # expected 1.0 but may yield 0.0 due to rounding
  ```

- **Catastrophic cancellation:** Subtracting nearly equal numbers can eliminate significant digits; rewrite expressions to avoid this when possible. For example, computing `sqrt(x+1) - sqrt(x)` for large x loses precision; instead use an algebraically equivalent stable form:

  ```python
  import math
  x = 1e10
  print(math.sqrt(x+1) - math.sqrt(x))            # unstable
  print(1.0 / (math.sqrt(x+1) + math.sqrt(x)))    # stable
  ```

- **NaN/Inf:** Check for NaN or infinite values during preprocessing — they break most algorithms. Use `np.isnan`/`np.isfinite` to detect and either impute or drop problematic entries.

## Numerical Stability in ML Algorithms
- Compute losses from raw logits when possible (e.g., stable softmax + cross-entropy) instead of converting to probabilities and then taking logs. Many libraries provide combined routines to avoid overflow/underflow:

  ```python
  # naive softmax + log loss (unstable)
  logits = np.array([1000, 1000])
  probs = np.exp(logits) / np.sum(np.exp(logits))  # overflow
  # use logsumexp trick instead
  from scipy.special import logsumexp
  log_probs = logits - logsumexp(logits)
  ```

- Use numerically stable functions from libraries (e.g., `scipy.special.logsumexp`, `numpy.matmul`/`@` for matrix mult). These routines handle edge cases and are implemented in C for performance.

## Memory & Type Choices
- For very large datasets or GPU training, prefer `float32`; for small numerical experiments or high-precision needs, use `float64`.
- When converting types, do so explicitly and consistently (e.g., `X.astype(np.float32)`) and ensure downstream code expects that dtype.

## Handling Large or Small Values
- Standardize or normalize features to avoid extremely large or tiny values that can harm optimization.
- When computing exponentials, clip inputs (or use stable implementations) to avoid overflow.

## Summary Checklist
- Ensure the correct dtype (`float32` vs `float64`) for your task.
- Check for NaN/Inf and handle missing values before training.
- Prefer stable numerical routines (log-sum-exp, stable softmax) for probability computations.
- Standardize features when appropriate to improve numeric conditioning.

````
