# Wine-Quality-Check
🍷 Wine Quality Prediction – End-to-End Data Science Project
This project was developed as part of my internship at Meta Courses, where I built a complete machine learning pipeline to predict the quality of red and white wines using physicochemical properties. The project also includes an interactive Streamlit + Plotly dashboard for real-time predictions.

### 📌Problem Statement
Predict the quality of wine (score between 0 to 10) based on various chemical properties using machine learning models.

### 📁 Dataset
Source: UCI Wine Quality Dataset
Types: Red wine & White wine
Features include:
Fixed acidity, Volatile acidity, Citric acid
pH, Density, Alcohol, and more
Target: quality (integer score)

### 🔍 Exploratory Data Analysis (EDA)
Visualized distributions of wine quality
Compared red and white wine characteristics
Correlation heatmaps and feature importance analysis

### 🧹 Data Preprocessing
Handled missing values
Normalized numerical features
Converted quality into binary/multi-class classification

### 🤖 Model Building
Trained and compared multiple models:
✅ Logistic Regression
🌲 Random Forest Classifier (best performer)

### 📈 Model Evaluation
Accuracy
Confusion Matrix
Precision, Recall, F1-score

### 🌐 Dashboard (Streamlit + Plotly)
Developed an interactive web-based dashboard:
User inputs wine features
Displays predicted wine quality
Dynamic visualizations with Plotly

### 🛠 Tech Stack
Language: Python
Libraries: Pandas, NumPy, Scikit-learn, Streamlit, Plotly, Seaborn, Matplotlib

### 🎓 Internship Info
📍 Organization: Meta Courses
🎯 Program: Data Science Internship

### 📂 Project Structure
Copy
Edit
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── notebooks/
│   └── EDA and model building
├── app.py
├── requirements.txt
└── README.md

### 📌 How to Use
Clone the repo
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the dashboard:
bash
Copy
Edit
streamlit run app.py
