# Decision tree classifier for Iris dataset

## What this project does

This beginner project trains and inspects a single Decision Tree classifier on the classic Iris dataset. The goal is to learn how a tree splits examples using four measurements (sepal length, sepal width, petal length, petal width) and to compare a deliberately shallow tree with an unconstrained tree to show how complexity affects generalisation. The project reports accuracy, per-class precision and recall on a held-out test split and prints human-readable decision rules and the path taken by a sample.

## Data

The project uses the built-in Iris dataset from scikit-learn. This dataset contains 150 examples, three class labels (setosa, versicolor, virginica) and four numeric measurements. Because this is a small teaching dataset, it is not a field-ready species identification product; it is used here solely for pedagogy.

## How to build it

1. Create a Python 3.11+ environment and install the requirements listed below.
2. Save project.py and test_project.py from this repository into a folder.
3. Ensure you have pytest available for running tests.
4. No external data downloads are necessary; the code uses sklearn's built-in Iris loader.

## How to run

Run the script directly with Python. It will print a JSON object with held-out metrics and a short textual explanation of one sample's path through the tree.

Example:

python project.py

## How to test

Run the automated tests with pytest in the same folder:

pytest -q

The tests check that the pipeline returns expected keys, that numeric metrics are in valid ranges, that a conservative accuracy floor is met on the test split, and that the main metric is deterministic for a fixed seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 0.9333 |
| accuracy_shallow | 0.8889 |
| precision_macro | 0.9444 |
| recall_macro | 0.9333 |
| explanation | The exported tree rules (excerpt):
|--- petal length (cm) <= 2.45
|   |--- class: 0
|--- petal length (cm) >  2.45
|   |--- petal width (cm) <= 1.55
|   |   |--- petal length (cm) <= 4.95
|   |   |   |--- class: 1
|   |   |--- petal length (cm) >  4.95
|   |   |   |--- class: 2
|   |--- petal width (cm) >  1.55
|   |   |--- petal width (cm) <= 1.70
|   |   |   |--- sepal width (cm) <= 2.85
|   |   |   |   |--- class: 1
|   |   |   |--- sepal width (cm) >  2.85
|   |   |   |   |--- class: 2
|   |   |--- petal width (cm) >  1.70
|   |   |   |--- petal length (cm) <= 4.85
|   |   |   |   |--- sepal width (cm) <= 3.00
|   |   |   |   |   |--- class: 2
|   |   |   |   |--- sepal width (cm) >  3.00
|   |   |   |   |   |--- class: 1
|   |   |   |--- petal length (cm) >  4.85
|   |   |   |   |--- class: 2

Sample index: 107 took path nodes: 0,2,6,10,14.
See the rules above for the numeric splits that led to the leaf. |
| precision_class_0 | 1 |
| precision_class_1 | 1 |
| precision_class_2 | 0.8333 |
| recall_class_0 | 1 |
| recall_class_1 | 0.8000 |
| recall_class_2 | 1 |

Also returned (see `python project.py` output): `confusion_matrix` (3 items).

> **Read the `precision_class_0`, `precision_class_1`, `recall_class_0`, `recall_class_2` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

Iris is a small, curated dataset intended for teaching. Results on Iris do not translate directly to robust species identification in the field. Decision trees can overfit when unconstrained; the shallow vs unconstrained comparison in this project demonstrates that behaviour. This project uses a single modelling approach (DecisionTreeClassifier) and numeric features only.

## Next steps

Possible extensions: inspect trees with different pruning parameters, cross-validate depth to pick a complexity that generalises, or replace the DecisionTreeClassifier with ensemble methods to study variance and bias trade-offs.

---

**Full case study, build steps and evaluation checks:** [Decision tree classifier for Iris dataset on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0004/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
