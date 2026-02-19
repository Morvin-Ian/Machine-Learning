**Fairness in ML — A Practical Intro**

Why fairness matters
- Models can unintentionally treat groups differently (by gender, race, age, etc.) because of biased data, proxies, or modeling choices.
- Fairness is important for ethics, legal compliance, and user trust.

Sources of bias
- Historical bias: historical data reflects unfair practices.
- Sampling bias: training data isn't representative of the population.
- Label bias: labels reflect human or process bias.
- Proxy features: innocuous features correlate with protected attributes.

Common fairness metrics (simple view)
- Demographic Parity: positive prediction rate should be equal across groups.
- Equalized Odds: true positive and false positive rates should be similar across groups.
- Predictive Parity: positive predictive value should be similar across groups.

Trade-offs
- You often cannot optimize all metrics at once; improving one can worsen another.
- Fairness may trade off with overall accuracy — quantify and make decisions explicit.

Mitigation strategies
- Pre-processing: modify data (reweighting, resampling) to reduce bias before training.
- In-processing: change learning algorithm to include fairness constraints or regularizers.
- Post-processing: adjust model outputs (thresholds or calibration) to improve fairness.

**Simple Python example using Fairlearn:**

```python
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import train_test_split
import numpy as np

# synthetic data: feature X, label y, sensitive attribute s (0/1)
X = np.random.randn(1000, 2)
y = (X[:,0] + X[:,1] > 0).astype(int)
s = (np.random.rand(1000) < 0.5).astype(int)

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, s, test_size=0.2, random_state=42)

base_clf = LogisticRegression(solver='liblinear')
constraint = DemographicParity()
mitigator = ExponentiatedGradient(base_clf, constraint)
mitigator.fit(X_train, y_train, sensitive_features=s_train)

preds = mitigator.predict(X_test)

# evaluate accuracy and group rates
from sklearn.metrics import accuracy_score, confusion_matrix
print('Overall accuracy', accuracy_score(y_test, preds))
for group in [0,1]:
    idx = s_test == group
    print(f'Group {group} positive rate', preds[idx].mean())
```

This shows how to enforce Demographic Parity by wrapping a classifier. The `fairlearn` library also offers many other metrics and mitigation algorithms.

Practical steps for teams
1. Identify protected groups relevant to your product and legal/regulatory context.
2. Define measurable fairness goals and metrics.
3. Test models on stratified holdout sets and compute chosen metrics.
4. If issues arise, try a mitigation technique and re-evaluate for unintended harms.
5. Monitor fairness over time — data drift can reintroduce bias.

Tools
- `AIF360`, `Fairlearn`, `What-If Tool` (TensorBoard), and built-in tooling in cloud providers can help measure and mitigate bias.

Ethics & governance
- Keep human oversight, document choices, and maintain clear audit logs of data and model decisions.
- Engage stakeholders and affected communities when possible.
