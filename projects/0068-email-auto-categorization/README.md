# Email auto-categorization

## What this project does

This project builds a small, deterministic text-classification pipeline that assigns incoming messages to user-defined folders. It simulates email threads and message text, preserves complete threads during splitting, trains a TF-IDF + Logistic Regression model, and marks low-confidence predictions as "uncertain" rather than forcing a folder label. The pipeline emphasises reversible suggestions: predicted labels are produced alongside a confidence score and a flag indicating uncertainty so human review is straightforward.

## Data

The data is synthetic but designed to resemble short email messages across realistic folder labels: Work, Personal, Promotions and Spam. Thread structure is simulated so that related messages remain grouped and all identifiers are removed. This is a small dataset intended for teaching: it uses in-memory generated messages from simple templates and topic-specific vocabularies.

## How to build it

1. Prepare the data — generate synthetic messages and thread ids, strip identifiers and split complete threads together.
2. Build the experiment — vectorise message text with TF-IDF and fit a Logistic Regression classifier, using a probability threshold to create an "uncertain" category.
3. Evaluate the result — compute held-out accuracy and per-folder recall; list a few high-impact misclassifications for inspection so reviewers can prioritise corrections.
4. Define a reviewable output — emit suggested labels with confidence and an "uncertain" flag so labels can be applied reversibly by a reviewer.

## How to run

Run the project module. The command prints a JSON dictionary of metrics and sample suggestions. Example (from a terminal):

python project.py

No external data or network access is required.

## How to test

Run pytest in the project directory. The tests verify deterministic behaviour, that expected metric keys exist, that metrics lie in valid ranges, and that the main held-out accuracy meets a conservative floor.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| accuracy | 0.9781 |
| macro_recall | 0.9703 |
| n_test_messages | 183 |
| n_train_messages | 404 |

Also returned (see `python project.py` output): `per_folder_recall` (object), `sample_high_conf_misclassifications` (0 items), `some_suggestions` (6 items).

## Limitations

This is a small teaching dataset generated synthetically from templates and topic word lists; it does not reflect the full diversity, privacy issues or adversarial content of real mailboxes. The model is simple (TF-IDF + Logistic Regression) and does not handle attachments, HTML, or entity-level redaction. The uncertain category is based on a single probability threshold; in production that threshold should be calibrated and reviewers should inspect edge cases.

## Next steps

Possible next steps include: use larger, consented corpora; add phrase or topic models; incorporate thread context beyond single-message text; calibrate uncertainty with validation data; and implement a lightweight reviewer UI that applies reversible labels.

---

**Full case study, build steps and evaluation checks:** [Email auto-categorization on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0068/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
