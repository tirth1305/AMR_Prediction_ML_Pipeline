# 🧬 AMR Prediction using Machine Learning

This project predicts antimicrobial resistance (AMR) using machine learning models built from genomic k-mer features.  
Achieved **~80% accuracy** using Random Forest and Stacking Ensemble (RF + XGBoost + Logistic Regression).

## 📂 Files
- `data/` → CSV files (train/test features & labels)
- `scripts/train_model.py` → Model training and evaluation script
- `requirements.txt` → Dependencies

## 🚀 Run
```bash
pip install -r requirements.txt
python scripts/train_model.py
