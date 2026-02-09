# Machine Learning Learning Path

A comprehensive, structured curriculum for learning machine learning from fundamentals to advanced techniques.

## 📚 Project Structure

This repository is organized into a clear learning progression with three main sections:

### [01-Fundamentals](./01-fundamentals)
Core mathematical concepts and algorithms essential for all machine learning work.

- **Gradient Descent** - The optimization algorithm that powers modern machine learning

### [02-Supervised Learning](./02-supervised-learning)
Algorithms that learn from labeled training data to make predictions.

- **Linear Regression** - Predicting continuous values
- **Logistic Regression** - Binary and multi-class classification
- **Classification** - Advanced classification techniques

### [03-Unsupervised Learning](./03-unsupervised-learning)
Algorithms that discover patterns in unlabeled data.

- Clustering techniques
- Dimensionality reduction
- Feature learning

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
   - Understand optimization and gradient descent

2. **Then explore**: [02-Supervised Learning](./02-supervised-learning)
   - Begin with linear regression for regression basics
   - Progress to logistic regression for classification
   - Advance to general classification techniques

3. **Finally master**: [03-Unsupervised Learning](./03-unsupervised-learning)
   - Apply knowledge to discover patterns in unlabeled data
   - Explore clustering and dimensionality reduction

## 📋 Dependencies

See `pyproject.toml` for full list. Key packages include:

- **numpy** - Numerical computing
- **pandas** - Data manipulation
- **matplotlib & plotly** - Data visualization
- **tensorflow & keras** - Deep learning
- **scikit-learn** (via google-ml-edu) - Machine learning algorithms

## 📖 How to Use This Repository

Each section contains:
- **notes.md** - Detailed explanations and theory
- **code files** - Practical implementations and examples
- **README.md** - Section-specific guidance and prerequisites

Start with the notes to understand the theory, then explore the code implementations.

## 🔗 File Structure Overview

```
machine-learning/
├── README.md                          (This file)
├── pyproject.toml                     (Project configuration)
│
├── 01-fundamentals/
│   ├── README.md
│   └── gradient-descent/
│       ├── notes.md
│       └── [implementation files]
│
├── 02-supervised-learning/
│   ├── README.md
│   ├── linear-regression/
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── notes.md
│   │   └── README.md
│   ├── logistic-regression/
│   │   ├── notes.md
│   │   └── README.md
│   └── classification/
│       ├── notes.md
│       └── README.md
│
└── 03-unsupervised-learning/
    ├── README.md
    └── notes.md
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

Start with [01-fundamentals](./01-fundamentals) and follow the learning path outlined above.
