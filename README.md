# Career Recommendation ML Model

This repository contains an ML pipeline to recommend careers based on **education, interest, skill, and personality** attributes. It is intended primarily for **educational and prototyping purposes** as part of a chatbot or career guidance tool.

---

## Dataset

- File: `data/realistic_career_dataset_22000_rows.csv`
- Contains 22,000 synthetically generated rows.
- Careers mapped via empirically weighted attribute distributions inspired by O*NET and labor market research.
- Columns: `career`, `education`, `interest`, `skill`, `personality`.

---

## Model & Performance

- Model: Random Forest Classifier with 200 trees.
- Feature encoding: One-hot encoding for input features.
- Target encoding: Label encoding for career classes.
- Performance on 20% held-out test set:
  - **Accuracy:** ~0.81 (varies depending on train/test split and random seed)
  - **Macro F1 score:** Approximate performance visible in classification report inside the notebook.
  - Confusion matrix and feature importance visualizations are available.

---

## Usage

1. **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

2. **Train the model (optional):**

    ```
    python src/train_model.py
    ```

3. **Make predictions:**

    ```
    python src/predict.py
    ```


## Limitations & Disclaimers

> ⚠️ **Important:** This model is trained on a synthetic dataset created from research-inspired attribute mappings. It is **not** based on actual user survey or longitudinal career outcome data.  
> Use this as a **demonstration or prototype** only. Consult professional career counselors and real user data for production-grade recommendations.

---


---

## Contributing

Contributions and improvements are welcome!

- Submit issues if you find bugs or want to propose enhancements.
- Pull requests for dataset extensions, real data integration, or model improvement encouraged.
- Please respect data provenance and user privacy.

---

## Contact

Anshika

---

**Example disclaimer to show prominently:**

>This tool is for educational and exploratory use only, and should not replace professional career counseling or advice.


