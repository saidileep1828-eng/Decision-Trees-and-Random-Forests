# Decision Trees and Random Forests

This workspace contains a Python script to train and evaluate a Decision Tree and a Random Forest classifier using the Heart Disease dataset.

## Files

- `decision_trees_random_forests.py`: full implementation of the task
- `decision_tree.png`: generated decision tree visualization (after running)
- `feature_importance.png`: generated feature importance plot (after running)

## Requirements

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn graphviz
```

## Usage

1. Place the `heart.csv` dataset in this folder.
2. Run:

```bash
python decision_trees_random_forests.py
```

The script will:

- print dataset overview
- train a decision tree and a random forest
- compare accuracy and overfitting
- save `decision_tree.png` and `feature_importance.png`
