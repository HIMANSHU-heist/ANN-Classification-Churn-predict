# 🧠 Customer Churn Prediction using Artificial Neural Network

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://ann-classification-churn-predictionbyhima.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **Will this bank customer stay or leave?** — An end-to-end ANN classification system that predicts customer churn before it happens.

---

## 🚀 Live Demo

🔗 **[Try it Live → ann-classification-churn-predictionbyhima.streamlit.app](https://ann-classification-churn-predictionbyhima.streamlit.app/)**

---

## 📌 Problem Statement

Banks lose crores every year to customer churn. Identifying **who will leave before they actually do** allows businesses to take proactive retention actions. This project builds an end-to-end ANN model that takes customer data as input and predicts the probability of churn.

---

## 🏗️ Project Architecture

```
Input Features (11)
        ↓
Feature Engineering
(One Hot Encoding + StandardScaler)
        ↓
Dense(64, ReLU)  ← Hidden Layer 1
        ↓
Dense(32, ReLU)  ← Hidden Layer 2
        ↓
Dense(1, Sigmoid) ← Output Layer
        ↓
Churn Probability (0 to 1)
```

---

## 📊 Dataset

- **Source:** [Churn Modelling Dataset — Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Size:** 10,000 bank customer records
- **Features:** Credit Score, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Target:** `Exited` (0 = Stayed, 1 = Left)

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn (StandardScaler, LabelEncoder) |
| Deployment | Streamlit |
| Model Saving | .h5 + Pickle |

---

## 🔬 What Happens Under the Hood

### 1. Data Preprocessing
```python
# One Hot Encoding for Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Label Encoding for Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```

### 2. ANN Architecture
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### 3. Training
```python
model.fit(X_train, y_train,
          epochs=100,
          batch_size=32,
          validation_data=(X_test, y_test))
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Validation Accuracy | ~86% |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Epochs | 100 |

---

## 🛠️ How to Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/ann-churn-prediction
cd ann-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
ann-churn-prediction/
│
├── app.py                  # Streamlit web app
├── train.py                # Model training script
├── model.h5                # Saved ANN model
├── scaler.pkl              # Saved StandardScaler
├── label_encoder.pkl       # Saved LabelEncoder
├── Churn_Modelling.csv     # Dataset
└── requirements.txt        # Dependencies
```

---

## 🧠 Key Learnings

- How ANN forward and backward propagation works
- Importance of feature scaling for neural networks
- One Hot Encoding vs Label Encoding — when to use what
- Sigmoid activation for binary classification
- Saving and loading models with Keras + Pickle

---

## 👨‍💻 Author

**Himanshu Bendale**
- 🎓 B.E. AI & DS — Mumbai University
- 🔗 [GitHub](https://github.com/https://github.com/HIMANSHU-heist)
- 💼 [LinkedIn](https://www.linkedin.com/in/himanshu-s-bendale-a695aa364?utm_source=share_via&utm_content=profile&utm_medium=member_android)

---

## 🗺️ Roadmap

```
✅ ANN — Churn Prediction (deployed)
✅ RNN — Sentiment Analysis (deployed)
✅ LSTM — Next Word Predictor (deployed)
🔜 Transformers — Attention Mechanism
🔜 LLM Fine-Tuning — LoRA/QLoRA on Llama 3
```

> *"The grind is on."* 💪
