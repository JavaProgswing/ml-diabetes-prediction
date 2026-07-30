# Diabetes Prediction

A beginner binary-classification project on the Pima Indians diabetes dataset. It
predicts diabetes from eight clinical measurements with logistic regression. The
point of interest is that several columns use `0` to mean "missing".

This project is educational only. It is not a medical diagnostic tool.

## Dataset

Download Kaggle's [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
and place `diabetes.csv` at:

```text
data/raw/diabetes.csv
```

With the Kaggle CLI:

```bash
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data/raw --unzip
```

The file has 768 rows. In `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`,
and `BMI` a value of `0` is physically impossible and really means the reading is
missing, so those zeros are converted to missing values and imputed with the
median. About 35% of patients are diabetic.

## Workflow

1. Load the raw CSV and remove exact duplicate rows.
2. Replace impossible zeros with missing values in the five affected columns.
3. Make a stratified 80/20 train/test split.
4. Impute missing values with the median and standardize - inside the pipeline,
   so the median is computed on training data only.
5. Train logistic regression.
6. Compare against a majority-class dummy model.
7. Evaluate with accuracy, precision, recall, F1, a confusion matrix, and
   five-fold stratified cross-validation.

## Results

| Measurement | Result |
|---|---:|
| Dummy test accuracy | 64.9% |
| Logistic regression test accuracy | 70.8% |
| Test recall (diabetics found) | 0.50 |
| CV mean accuracy | 78.8% |

Accuracy beats the baseline, but recall of 0.50 means the model misses half the
diabetic patients - the metric that matters most in a screening context. Raising
recall (class weighting or a lower decision threshold) is the natural next
experiment. Note the single test split (0.71) reads lower than cross-validation
(0.79), so the CV number is the more reliable estimate.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

The confusion matrix is written to `reports/figures/confusion_matrix.png`.

## Layout

```text
data/raw/         downloaded dataset (git-ignored - see Dataset above)
src/preprocess.py loading, cleaning, zero-as-missing, imputing scaler
src/train.py      pipeline, training loop, evaluation, figure
src/evaluate.py   classification metric helpers
main.py           entry point
```

## Why this project

Inspired by classmate medical-classification projects (e.g. PranavSingla122's
diagnostic notebooks, adityaxdubey's health models). It sits next to the
heart-disease classifier but adds the zero-as-missing cleaning step and median
imputation.
