**Automated ML (AutoML) — Simple Explanation**

What is AutoML?
- AutoML automates parts of the machine learning workflow: model selection, hyperparameter tuning, and sometimes feature engineering or neural architecture search.
- The goal is to make ML accessible and faster to iterate.

Main components
- Search space: the set of models, preprocessing steps, and hyperparameters to try.
- Search strategy: how to explore the search space (random search, Bayesian optimization, evolutionary search).
- Evaluation strategy: how candidate models are measured (cross-validation, holdout sets, cost-aware metrics).

Kinds of automation
- Hyperparameter tuning: automates finding good settings for a chosen model.
- Pipeline search: composes preprocessing + model (e.g., TPOT, Auto-sklearn).
- Neural Architecture Search (NAS): designs neural network topologies automatically.
- End-to-end AutoML: handles feature preprocessing, model selection, and ensembling (e.g., H2O AutoML, Google AutoML, AutoGluon).

Pros
- Faster prototyping and baseline models with little manual effort.
- Good for teams without deep model engineering expertise.

Cons and caveats
- Can be compute-heavy and costly for large search spaces.
- May yield overly complex models that are hard to maintain.
- Still requires human oversight for data quality, fairness, and production-readiness.

When to use AutoML
- Use it for quick baselines, proof-of-concepts, or when feature engineering is standard.
- Avoid for highly custom architectures or when interpretability and strict constraints are required.

Practical tips
- Limit search space to reasonable model families.
- Use budgeted runs (max time or trials).
- Validate final model with domain-specific tests before production.

Tooling examples
- `Auto-sklearn`, `TPOT`, `H2O AutoML`, `AutoGluon`, cloud AutoML offerings (Google, AWS, Azure).

**Example using auto-sklearn**:

```python
import autosklearn.classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,  # seconds
    per_run_time_limit=30,
    ensemble_size=50,
)
automl.fit(X_train, y_train)
print(automl.leaderboard())
print('Test accuracy', automl.score(X_test, y_test))
``` 

Automated pipelines like TPOT work similarly but use genetic programming to evolve scikit-learn pipelines.