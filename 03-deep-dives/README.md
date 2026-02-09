# Deep Dives

In-depth mathematical concepts and detailed explanations of core machine learning algorithms.

## Purpose

This section contains **advanced theoretical content** that explains the "how" and "why" behind machine learning algorithms. Reference these topics when you want a deeper understanding of the foundations.

## Topics Covered

### [Working with Data](./working-with-data/README.md)
Understanding and preparing data for machine learning.
- **Data types**: Numerical (continuous/discrete) and categorical (nominal/ordinal)
- **Feature scaling**: Normalization vs standardization
- **Encoding**: Label, one-hot, and target encoding
- **Dataset splitting**: Train/test/validation strategies
- **Generalization**: What it means and why it matters
- **Overfitting & Underfitting**: Signs, causes, and solutions
- **Bias-Variance Tradeoff**: The fundamental ML tradeoff
- **Learning Curves**: Diagnosing model problems
- **Practical workflow**: Complete data preparation pipeline

---

### [Gradient Descent](./gradient-descent/notes.md)
The optimization algorithm that powers ML.
- Intuitive "ball rolling downhill" analogy
- Mathematical derivation step-by-step
- Variants: Batch, Stochastic, Mini-batch
- Learning rate and convergence
- Advanced optimizers: Momentum, RMSprop, Adam

---

### [Multiple Linear Regression & Vectorization](./multiple-linear-regression/notes.md)
Handling multiple features efficiently.
- From one feature to many: notation and equations
- **Vectorization**: Why loops are slow, how NumPy is fast
- Matrix operations for predictions and gradients
- Complete NumPy implementation from scratch
- Feature scaling and normalization
- NumPy operations reference

---

## When to Read These

| Scenario | Topic |
|----------|-------|
| "What type of data do I have?" | **Working with Data** |
| "How do I encode categorical features?" | **Working with Data** |
| "My model overfits/underfits" | **Working with Data** (Generalization) |
| "What's the bias-variance tradeoff?" | **Working with Data** (Bias-Variance) |
| "How does training actually work?" | **Gradient Descent** |
| "How do I handle multiple features?" | **Multiple Linear Regression** |
| "Why is my code so slow?" | **Vectorization** section |
| Model not converging | Gradient Descent troubleshooting |
| Want to understand optimizers | Advanced Optimizers section |

## Structure

```
03-deep-dives/
├── README.md                          (This file)
├── working-with-data/
│   ├── README.md                      (Section overview)
│   ├── notes.md                       (Data types, encoding, splitting)
│   └── generalization-overfitting.md  (Overfitting, bias-variance, learning curves)
├── gradient-descent/
│   └── notes.md                       (400+ lines)
└── multiple-linear-regression/
    └── notes.md                       (Vectorization & NumPy)
```

## Prerequisites

These topics assume familiarity with:
- Basic calculus (derivatives)
- Linear algebra basics (vectors, matrices)
- Python programming

## Suggested Reading Order

1. Read supervised learning topics first (`01-supervised-learning`)
2. When you encounter gradient descent, come here for the full explanation
3. Read **Multiple Linear Regression** when working with multiple features
4. Return to supervised/unsupervised learning with deeper understanding

## Coming Soon

Future deep dives planned:
- Regularization (L1, L2, Elastic Net)
- Bias-Variance Tradeoff
- Cross-Validation
- Feature Engineering
- Neural Network Backpropagation

---

**Note:** These are reference materials, not the starting point. Begin with [01-supervised-learning](../01-supervised-learning/README.md).
