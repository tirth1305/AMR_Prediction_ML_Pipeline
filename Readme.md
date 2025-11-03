# 🧬 Machine Learning-Based Prediction of Antimicrobial Resistance (AMR)

This repository contains a machine learning pipeline for predicting antimicrobial resistance (AMR) in *Acinetobacter baumannii* isolates using k-mer based genomic features.  
The project demonstrates data preprocessing, feature selection, model training, and ensemble learning for accurate AMR classification.

---

## 🚀 Project Overview

Antimicrobial resistance (AMR) poses a major threat to global health.  
This study applies machine learning to predict resistance phenotypes directly from genomic k-mer features, offering a faster alternative to traditional phenotype-based testing.

The pipeline uses:
- **Random Forest** for baseline model training
- **Stacking Ensemble (RF + XGBoost + Logistic Regression)** for improved performance
- **Mutual Information-based Feature Selection** to reduce dimensionality and overfitting

---

## 🧩 Files Included

| File | Description |
|------|--------------|
| `X_train_sel.csv` | Training feature matrix (selected k-mers) |
| `X_test_sel.csv` | Test feature matrix (selected k-mers) |
| `y_train.csv` | Training labels (resistant/susceptible) |
| `y_test.csv` | Test labels (resistant/susceptible) |
| `amr_model.py` | Python script for model training and evaluation |
| `README.md` | Project documentation |

---

## 🧠 Methodology

### 1. **Data Preparation**
- Genomic data from *Acinetobacter baumannii* isolates were processed to extract **k-mer features**.
- Features were filtered using **Mutual Information** to retain the most informative 750–1000 k-mers.

### 2. **Model Training**
- **Random Forest Classifier** trained with balanced class weights:
  ```python
  RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
  ```
- **Stacking Ensemble** combining:
  - Random Forest
  - XGBoost
  - Logistic Regression (as meta-learner)

### 3. **Evaluation Metrics**
- Accuracy
- ROC-AUC
- Classification Report
- Confusion Matrix

---

## 📊 Results

| Metric | Random Forest | Stacking Ensemble |
|--------|----------------|------------------|
| Accuracy | ~78–80% | Slightly higher |
| ROC-AUC | Good separation | Improved robustness |
| Class Balance | Handled using `class_weight='balanced'` | Effective with stacking |

The ensemble model provided stable and improved results compared to single classifiers.

---

## 🧰 Dependencies

Install required libraries:
```bash
pip install pandas scikit-learn xgboost
```

---

## 🧪 Run the Model

1. Clone the repository:
   ```bash
   git clone https://github.com/tirth1305/AMR_Prediction_ML_Pipeline.git
   cd AMR_Prediction_ML_Pipeline
   ```

2. Run the model script:
   ```bash
   python amr_model.py
   ```

3. View the output metrics in the terminal.

---

## 🧬 Research Context

This work is inspired by:
> *Machine learning and feature extraction for rapid antimicrobial resistance prediction of Acinetobacter baumannii from whole-genome sequencing data* (Frontiers in Microbiology, 2024)

Dataset source:  
**BV-BRC (Bacterial and Viral Bioinformatics Resource Center)**

---

## 👨‍💻 Author

**Tirth Patel**  
Department of Bioinformatics, Marwadi University  
💼 Passionate about Genomics, Machine Learning, and Antimicrobial Resistance Research  

---

## 📜 License

This project is open-source under the MIT License.
