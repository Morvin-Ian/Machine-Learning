# Supervised Learning — Start Here! 🚀

Machine learning algorithms that learn from labeled training data. In other words, the algorithm is given examples where the correct answer is already known and it tries to infer a rule that will generalize to unseen examples.

## Why Start Here?

Supervised learning is the most intuitive introduction to ML:
- You have **labeled data** (inputs with known correct outputs) which makes the learning task well defined.
- The goal is clear: **predict the label for new data**, whether that label is a continuous quantity (regression) or a category (classification).
- Results are easy to evaluate with standard metrics like mean squared error or accuracy, so you can iterate quickly.

Moreover, many more advanced techniques (neural nets, gradient boosting, etc.) are just supervised learners at their core, so mastering these basics pays dividends across the field.

## Topics Covered

### 1. [Linear Regression](./linear-regression/notes.md) — **First Topic**
Predict continuous values. Perfect for understanding ML fundamentals.

- What is regression?
- Loss functions (MSE, MAE)
- How models learn (gradient descent basics)
- Multiple features

### 2. [Logistic Regression](./logistic-regression/notes.md)
Binary classification using probability.

- Sigmoid function
- Log loss / Cross-entropy
- Decision boundaries
- Regularization

### 3. [Classification](./classification/notes.md)
How to evaluate classification models.

- Confusion matrix
- Precision, Recall, F1-Score
- ROC curves and AUC
- Multi-class strategies

## Learning Order

```
Start here:
    ↓
┌─────────────────────┐
│  Linear Regression  │  ← Learn regression + how models train
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Logistic Regression │  ← Apply to classification
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Classification    │  ← Learn how to evaluate models
└──────────┬──────────┘
           ↓
    Continue to 02-unsupervised-learning
```

## Structure

```
01-supervised-learning/
├── README.md              (This file)
├── linear-regression/
│   ├── main.py           (Implementation)
│   ├── model.py          (Model code)
│   └── notes.md          (Theory + examples)
├── logistic-regression/
│   └── notes.md
└── classification/
    └── notes.md
```

## Prerequisites

- Basic Python programming
- High school math (algebra, basic statistics)

For deeper mathematical understanding, see [03-deep-dives](../03-deep-dives/README.md).

## Next Steps

After completing supervised learning:
- [02-unsupervised-learning](../02-unsupervised-learning) - Discover patterns in unlabeled data
- [03-deep-dives/gradient-descent](../03-deep-dives/gradient-descent/notes.md) - Deep dive into optimization
