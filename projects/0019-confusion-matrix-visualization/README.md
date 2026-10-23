# Confusion matrix explorer

## What this project does
This small project trains a multi-class classifier on the built-in Iris dataset and produces both count and row-normalised confusion matrices with explicit class ordering. It emphasises how counts and normalised rows answer different questions when class frequencies differ. The pipeline also provides a compact "class-error explorer" listing counts, per-class precision/recall, error rates and a few example mistakes from the held-out test split.

## Data
The project uses the built-in Iris dataset from scikit-learn (three classes, four numeric features). The data are not external — they are loaded from scikit-learn so the experiment is deterministic and reproducible.

## How to build it
1. Create a virtual environment (optional) and install dependencies: pip install -r requirements.txt
2. Inspect project.py to see the data split, model and metrics computed.
3. Run project.py directly (python project.py) to print the result JSON for a default seed.
4. Run the tests with pytest to verify determinism and minimum held-out performance.

## How to run
Run the script with Python 3.11+: python project.py. The script prints a JSON object with accuracy, macro precision/recall, confusion matrices (counts and row-normalised), a manual check for one class, and the class-error explorer.

## How to test
Run pytest in the repository root. The provided tests verify that run(seed) returns the expected keys, that numeric metrics are in sensible ranges, a conservative accuracy floor is met and that runs with the same seed are deterministic.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 0.8889 |
| precision_macro | 0.8981 |
| recall_macro | 0.8889 |

Also returned (see `python project.py` output): `confusion_counts` (3 items), `confusion_row_normalized` (3 items), `manual_check_class_0` (object), `class_error_explorer` (3 items).

## Limitations
This example uses the small, well-known Iris dataset which is not representative of many real-world problems (it is small, clean and balanced). The confusion matrices and metrics shown are therefore illustrative rather than proving real-world generalisation. The project does not include visual plotting of the confusion matrix (it outputs numeric matrices and explorer tables suitable for rendering elsewhere). The model choice (RandomForest) is simple and well-suited to this toy dataset; different datasets or cost structures may require different models and evaluation strategies.

## Next steps
- Add a visual rendering (heatmap) of the count and normalised confusion matrices in an environment that supports plotting.
- Try imbalanced or larger multi-class datasets (or create an imbalanced variant of Iris) to explore how counts vs normalised rows shift interpretation.
- Extend the explorer to show feature values or decision explanations for typical mistakes (for example, with SHAP or simple feature comparisons).

---

**Full case study, build steps and evaluation checks:** [Confusion matrix visualization on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0019/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
