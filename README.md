## BCI Systematic and Interpretable Feature Tuning (BCI-sift)
`BCI-sift` is a Python package for identifying the most relevant features in BCI tasks. It is described in the manuscript [BCI-sift: An automated feature selection toolbox for Brain Computer Interface applications](#). The packages implements diverse optimization algorithms to perform feature selection and can thereby enhance the decoding accuracy, reduce computational demands, and improve the interpretability of BCI systems.

The toolbox currently provides seven optimization strategies: 
- `RecursiveFeatureElimination`, which iteratively removes the least informative features based on model performance
- `EvolutionaryAlgorithms`, which optimize feature subsets using population-based evolutionary strategies
- `ParticleSwarmOptimization`, which explores the feature space using a swarm-based optimization approach
- `SimulatedAnnealing`, which performs probabilistic search by accepting both improving and, with decreasing likelihood, non-improving feature subsets
- `ContiguousExhaustiveSearch`, which evaluates all admissible contiguous feature subsets (e.g., continuous windows or rectangular regions)
- `StochasticHillClimbing`, which approximates contiguous search through iterative stochastic expansion of feature subsets
- `RandomSearch`, which samples feature subsets uniformly at random to provide a baseline comparison

## A. Installation

### Requirements

The package is tested with:

- Python `3.9`
- Linux

The core runtime dependencies are:

- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `pandas`
- `tqdm`

### Installation

Clone the repository and install it into a local Python environment.

```bash
git clone https://github.com/UMCU-RIBS/BCI-sift.git
cd BCI-sift
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

And install the dependencies: 

```bash
pip install "numpy>=1.25,<2.0" "pandas>=1.5,<3.0" "scipy>=1.11,<2.0" "scikit-learn>=1.3,<1.5" "matplotlib>=3.7,<4.0" "seaborn>=0.13,<0.14" "tqdm>=4.66,<5.0"
```

## B. Quick start

The example below demonstrates an example of recursive feature elimination used on synthetic input with the expected tensor shape `[samples, time, channels]`, see also `examples/minimal_RFE_example.py`.

```python
import sys
sys.path.insert(0, '.')
from optimizer import RecursiveFeatureElimination
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

X, y = make_classification(n_samples=150, n_features=50 * 20)# 150 trials, 50 time points, 20 channels
X = X.reshape(150, 50, 20)

#optimize time points first, and then channels 
estimator = Pipeline([('scaler', MinMaxScaler()), ('svc', SVC(kernel="linear"))])
rfe = RecursiveFeatureElimination(dimensions=(1,2), feature_space = "tabular", estimator=estimator, importance_getter = "named_steps.svc.coef_", verbose=True)
rfe.fit(X, y)
print(rfe.score_)
```

## C. Advanced example
`examples/advanced_bcisift_example.py` is an extensive example of using the BCI-sift toolbox, including loading the data, using nested cross-validation, running different optimization methods, 
and plotting and saving the results. The optimizers are run in a nested cross-validation scheme, where the inner loop is used for optimization and the outer loop is used for evaluating the 
performance of the optimized model on unseen data. The results are saved in a csv file for each subject, and include the best score, the best mask, and the number of selected channels for each method. 
Additionally, the importance and elimination plots are saved for each method.

`examples/advanced_bcisift_example_config.yml` includes all parameters used in this example.

## D. Project Structure

```text
BCI-sift/
├── README.md
├── LICENSE
├── .gitignore
├── dataset/
│   ├── __init__.py
│   ├── Dataset.py
│   ├── Epochs.py
│   ├── Events.py
│   └── utils.py
├── optimizer/
│   ├── __init__.py
│   ├── Base_Optimizer.py
│   ├── ContiguousExhaustiveSearch.py
│   ├── EvolutionaryAlgorithms.py
│   ├── ParticleSwarmOptimization.py
│   ├── RandomSearch.py
│   ├── RecursiveFeatureElimination.py
│   ├── SimulatedAnnealing.py
│   ├── StochasticHillClimbing.py
│   └── utils.py
├── examples/
│   ├── minimal_RFE_example.py
│   ├── advanced_bcisift_example.py
│   └── advanced_bcisift_example_config.yml
└── utils/
    ├── __init__.py
    ├── custom_estimator.py
    ├── grid_plots.py
    └── hp_tune.png
```

## E. Citation

If you use this repository, please cite the associated manuscript and the software repository.

```bibtex
@article{offenberg2026bcisift,
  author = {Offenberg, Elena C. and Keller, Dirk and Vansteensel, Mariska J. and Freudenburg, Zachary V and Ramsey, Nick F and Berezutskaya, Julia},
  title = {BCI-sift: An automated feature selection toolbox for Brain Computer Interface applications },
  year = {2026},
  journal = {To be added},
  doi = {To be added}
}

