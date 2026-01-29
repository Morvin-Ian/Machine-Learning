# Machine Learning Learning Path

A comprehensive, structured curriculum for learning machine learning from fundamentals to advanced techniques.

## 📚 Project Structure

This repository is organized into a clear learning progression with three main sections:

### [01-Fundamentals](./01-fundamentals)
Core mathematical concepts and algorithms essential for all machine learning work.

- **[Gradient Descent](./01-fundamentals/gradient-descent/notes.md)** - The optimization algorithm that powers modern machine learning
  - Intuitive explanation with visual analogies
  - Mathematical derivation and update rules
  - Variants: Batch, Stochastic, Mini-batch
  - Learning rate and convergence
  - Advanced optimizers: Momentum, Adam

### [02-Supervised Learning](./02-supervised-learning)
Algorithms that learn from labeled training data to make predictions.

- **[Linear Regression](./02-supervised-learning/linear-regression/notes.md)** - Predicting continuous values
- **[Logistic Regression](./02-supervised-learning/logistic-regression/notes.md)** - Binary and multi-class classification
- **[Classification](./02-supervised-learning/classification/notes.md)** - Evaluation metrics and thresholds

### [03-Unsupervised Learning](./03-unsupervised-learning)
Algorithms that discover patterns in unlabeled data.

- **[Clustering](./03-unsupervised-learning/clustering/notes.md)** - K-Means, Hierarchical, DBSCAN
- **[Dimensionality Reduction](./03-unsupervised-learning/dimensionality-reduction/notes.md)** - PCA, t-SNE
- **[Anomaly Detection](./03-unsupervised-learning/anomaly-detection/notes.md)** - Isolation Forest, One-Class SVM

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- pip or uv package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd machine-learning

# Install dependencies
pip install -r requirements.txt
# or with uv:
uv sync
```

### Recommended Learning Order

1. **Start here**: [01-Fundamentals](./01-fundamentals)
   - Build strong mathematical foundations
   - Master gradient descent optimization

2. **Then explore**: [02-Supervised Learning](./02-supervised-learning)
   - Begin with linear regression for regression basics
   - Progress to logistic regression for classification
   - Advance to classification metrics and evaluation

3. **Finally master**: [03-Unsupervised Learning](./03-unsupervised-learning)
   - Start with clustering (K-Means)
   - Learn dimensionality reduction (PCA)
   - Explore anomaly detection

## 📋 Dependencies

See `pyproject.toml` for full list. Key packages include:

- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **matplotlib & plotly** - Data visualization
- **scikit-learn** - Machine learning algorithms
- **tensorflow & keras** - Deep learning

## 📖 How to Use This Repository

Each section contains:
- **notes.md** - Detailed explanations, theory, and examples
- **code files** - Practical implementations
- **README.md** - Section-specific guidance and prerequisites

Start with the notes to understand the theory, then explore the code implementations.

## 🔗 File Structure Overview

```
machine-learning/
├── README.md                           (This file)
├── pyproject.toml                      (Project configuration)
│
├── 01-fundamentals/
│   ├── README.md
│   └── gradient-descent/
│       └── notes.md                    (Comprehensive GD guide)
│
├── 02-supervised-learning/
│   ├── README.md
│   ├── linear-regression/
│   │   ├── main.py
│   │   ├── model.py
│   │   └── notes.md
│   ├── logistic-regression/
│   │   └── notes.md
│   └── classification/
│       └── notes.md
│
└── 03-unsupervised-learning/
    ├── README.md
    ├── notes.md                        (Overview)
    ├── clustering/
    │   └── notes.md                    (K-Means, Hierarchical, DBSCAN)
    ├── dimensionality-reduction/
    │   └── notes.md                    (PCA, t-SNE)
    └── anomaly-detection/
        └── notes.md                    (Isolation Forest, etc.)
```

## 💡 Tips for Success

1. **Understand the theory** - Read the notes before running code
2. **Experiment** - Modify code examples and explore variations
3. **Practice** - Implement algorithms from scratch when possible
4. **Apply** - Find datasets and apply techniques to real problems
5. **Reference** - Bookmark official documentation for libraries used

## 🤝 Contributing

Feel free to improve this learning path! Suggestions are welcome.

## 📝 License

[Add your license information here]

---

**Happy Learning!** 🎓

Start with [01-fundamentals/gradient-descent](./01-fundamentals/gradient-descent/notes.md) and follow the learning path outlined above.
