'''
This is a minimal example of how to use the RecursiveFeatureElimination class on a BCI dataset.
'''
import sys
sys.path.insert(0, '.')
from optimizer import RecursiveFeatureElimination
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=100, n_features=8 * 4 * 100)
X = X.reshape((100, 8, 4, 100))

X, y = make_classification(n_samples=150, n_features=50 * 20)# 150 trials, 50 time points, 20 channels
X = X.reshape(150, 50, 20)

#optimize time points first, and then channels 
estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC(kernel="linear"))])
rfe = RecursiveFeatureElimination(dimensions=(1,2), feature_space = "tabular", estimator=estimator, importance_getter = "named_steps.svc.coef_", verbose=True)
rfe.fit(X, y)
print(rfe.score_)