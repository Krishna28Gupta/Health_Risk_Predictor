# 🩺 Health Risk Prediction Portal

This project is a **Machine Learning–powered Health Risk Prediction System** built using **Streamlit**.  
It predicts the risk of **Lung Cancer, Stroke, Diabetes, and Heart Disease** based on user-entered health data.

The models are trained using popular medical datasets and saved as `.pkl` files for fast inference.

---

## 🚀 Features
- ✔ Predict **4 major health risks**  
- ✔ User-friendly **Streamlit web interface**  
- ✔ Pre-trained ML models using:
  - Random Forest
- ✔ Clean modular code  
- ✔ Real-time predictions  
- ✔ Completely open-source  

---

## 📂 Project Structure

```
Health_Risk_Predictor/
│
├── data/
│   ├── diabetes_prediction_dataset.csv
│   ├── healthcare-dataset-stroke-data.csv
│   ├── heart (1).csv
│   └── survey lung cancer.csv
│
├── models/
│   ├── diabetes_prediction_model.pkl
│   ├── HeartDisease_prediction_model.pkl
│   ├── LungCancer_prediction_model.pkl
│   ├── stroke_prediction_model.pkl
│   └── label_encoders.pkl
│
├── notebooks/
│   ├── diabetes.ipynb
│   ├── HeartDisease.ipynb
│   ├── Lung_Cancer.ipynb
│   └── stroke_pred.ipynb
│
├── app/
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## 🛠 Tech Stack

- **Python 3.x**
- **Streamlit**
- **Pandas, NumPy**
- **Scikit-Learn**
- **Joblib**

---

## 📦 Installation (Local Setup)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Health_Risk_Predictor.git
cd Health_Risk_Predictor
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app/main.py
```

---

## 📊 Machine Learning Models

| Disease        | Model Used | Accuracy | File |
|----------------|------------|----------|-----------------------------|
| Lung Cancer    | Random Forest | 89% | `LungCancer_prediction_model.pkl` |
| Stroke         | Random Forest | 84% | `stroke_prediction_model.pkl` |
| Diabetes       | Random Forest | 97% | `diabetes_prediction_model.pkl` |
| Heart Disease  | Random Forest | 89% | `HeartDisease_prediction_model.pkl` |


---

## 🖼 Screenshots


### 🔹 Home Page
<img width="1920" height="1013" alt="Homepage png" src="https://github.com/user-attachments/assets/fe30d59d-893a-466e-b01f-2df2a755e8bd" />


### 🔹 Diabetes Form
<img width="1920" height="1019" alt="DiabetesRisk png" src="https://github.com/user-attachments/assets/c6446e6d-c2ab-442e-903c-8a51f9e4467a" />



---

## ⚠️ Disclaimer

This application provides **risk predictions only** based on statistical models trained on historical datasets.  
It **cannot replace professional medical diagnosis or treatment**.  
Consult a qualified healthcare professional for actual medical concerns.

---

## ⭐ Contribution

Feel free to open issues or submit pull requests to improve the project!

---

## 📧 Contact

**Krishna Gupta**  
Email: guptak143600@gmail.com 
GitHub: https://github.com/Krishna28Gupta



