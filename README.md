
<h1 align="center">🔬 Breast Cancer Detection using Decision Tree & Machine Learning 🧠</h1>
<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Matplotlib-Visualization-green?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Accuracy-99.12%25-success?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge" />
</p>

---

## 📌 **Project Overview**
This project focuses on predicting **Breast Cancer Diagnosis** (**Benign** or **Malignant**) using a **Decision Tree Classifier**.  
We trained the model using the **Breast Cancer dataset** and performed data preprocessing, visualization, hyperparameter tuning,  
model evaluation, and deployment using **Gradio** for real-time predictions.

---

## 🧠 **Technologies Used**
| Tool / Library | Purpose |
|---------------|------------------------|
| **Python** | Programming Language |
| **Pandas, NumPy** | Data Handling & Preprocessing |
| **Matplotlib, Seaborn, Plotly** | Data Visualization |
| **Scikit-learn** | ML Model Training & Evaluation |
| **Gradio** | Interactive Web App Deployment |
| **Google Colab** | Model Training & Testing |

---

## 📂 **Dataset Information**
- **Dataset Name:** Breast Cancer Dataset  
- **Source:** Uploaded CSV (`cleaned_breast_cancer dataset.csv`)
- **Target Column:** `diagnosis`  
    - `0` → Benign (Non-cancerous)  
    - `1` → Malignant (Cancerous)  
- **Shape:** **569 rows × 31 columns**

---

## 📊 **Project Workflow**
We followed a **step-by-step approach** to build a robust Breast Cancer Detection model:

---

### 🟢 **Step 1 — Import Required Libraries**  
Loaded all essential **Python libraries** for:
- Machine Learning 🧠  
- Data Visualization 📊  
- Model Evaluation ✅  

---

### 📂 **Step 2 — Load Dataset**  
- Loaded the **Breast Cancer dataset** (CSV file).  
- Explored dataset shape, columns, and structure.  
- Verified data consistency for better model performance.

---

### 🧩 **Step 3 — Data Preprocessing**  
- Removed **unnecessary columns** 🗑️  
- Converted **categorical labels → numerical labels**  
- Handled **missing values** (if any)  
- Scaled the features for better accuracy 📈  

---

### ✂️ **Step 4 — Train/Test Split**  
Divided the dataset into:
- **80% Training Data** 🏋️‍♂️  
- **20% Testing Data** 🧪  
Ensured a **stratified split** to maintain label balance.

---

### 🌳 **Step 5 — Model Training**  
- Trained a **Decision Tree Classifier** 🌲  
- Performed **hyperparameter tuning** using **GridSearchCV** 🔍  
- Selected the **best estimator** for maximum accuracy.  

---

### 📊 **Step 6 — Model Evaluation**  
Evaluated the model performance using multiple metrics:
- ✅ **Accuracy Score**
- 🧾 **Classification Report** (Precision, Recall, F1-score)  
- 🔁 **Confusion Matrix**  
- 📈 **ROC Curve & AUC Score**  

---

### 📈 **Step 7 — Visualization Dashboard**  
Created an **interactive visualization dashboard** using:
- **Matplotlib** & **Seaborn** 📊  
- **Plotly** for dynamic interactivity 🎨  

Visualizations include:
- ROC Curve 📉  
- Confusion Matrix 🟩  
- Feature Importance Ranking ⭐  

---

### 🔮 **Step 8 — User Input Prediction**  
Users can **predict breast cancer diagnosis** using:
- **Manual input** via terminal ✍️  
- **Uploading a `.txt` file** 📄  

The model predicts:
- 🟢 **Benign** *(Non-cancerous)*  
- 🔴 **Malignant** *(Cancerous)*  
- 📊 Displays **probability of malignancy** in %.  

---

### 🌐 **Step 9 — Gradio Web App** *(Optional)*  
An **interactive web interface** was created using **Gradio**:  
- Users can enter input values easily 🧑‍💻  
- Get instant predictions ⚡  
- Visualize the results intuitively 🎯  

---

### 🧪 **Step 10 — Model Testing on Unseen Data**  
- Verified model generalization on unseen samples.  
- Ensured **high accuracy** and **low overfitting**.  

---

### 📌 **Step 11 — Cross-Validation**  
- Performed **k-fold cross-validation** 🔄  
- Ensured stable and reliable model performance.  

---

### ⚡ **Step 12 — Hyperparameter Tuning**  
- Optimized key hyperparameters like:  
    - `max_depth`  
    - `min_samples_split`  
    - `criterion`  
- Improved **accuracy & precision** significantly.

---

### 🏆 **Step 13 — Evaluate the Best Decision Tree**  
- Compared initial vs tuned models.  
- Selected the **final best-performing model** 🥇.

---

### 🖼️ **Step 14 — Generate & Save Graphs**  
- Saved all visualizations as `.png` files 🖼️.  
- Added graphs to the **final project report** 📑.

---

### 🔁 **Step 15 — Model Comparison**  
Compared:
- Initial Decision Tree vs Tuned Decision Tree  
- Visualized improvements in metrics & accuracy 📊.

---

### 🧑‍💻 **Step 16 — User Input-Based Prediction**  
- Model accepts **user input** values interactively.  
- Predicts if the cancer is **Benign** or **Malignant**.  
- Displays the **probability of malignancy** 🔮.

---

### 📄 **Step 17 — Prediction via `.txt` File**  
- Upload a `.txt` file containing feature values.  
- The model reads the file and predicts diagnosis automatically ✅.

---

### 💾 **Step 18 — Export & Save Model Results**  
- Exported trained model using **Pickle** / **Joblib** 🗄️.  
- Stored results for future usage and deployment.

---

### 📝 **Step 19 — Final Project Reporting**  
- Prepared the **final project report** 📄  
- Included:
    - Methodology  
    - Dataset Details  
    - Model Insights  
    - Visualizations  
    - Evaluation Metrics  

---

## 🧪 **Model Performance**
| Metric | Score |
|--------|--------|
| **Training Accuracy** | **99.12%** |
| **Testing Accuracy** | **96.49%** |
| **AUC Score** | **0.9823** |
| **Cross-Validation** | **96.10% ± 1.8%** |

---

## ✍️ **Author**

**Priyanshu Taria**  
🎓 Graduate **B.Tech — Computer Science & Engineering**  
🏫 **Gandhi Institute of Engineering & Technology University (GIETU), Odisha**  
📧 **Email:** [priyanshutaria7327@gmail.com](mailto:priyanshutaria7327@gmail.com)  
🔗 **LinkedIn:** [www.linkedin.com/in/er-priyanshu-taria-055774283](https://www.linkedin.com/in/er-priyanshu-taria-055774283)

---
