##AI-Based Diabetes Prediction System

Machine Learning based diabetes prediction system developed using Random Forest Classifier and Streamlit.  
The system predicts whether a person is likely to have diabetes based on various health indicators and lifestyle-related factors.

---

 ##Features

- Diabetes prediction using health indicators
- Random Forest Classifier for prediction
- Streamlit-based interactive web application
- Real-time prediction and risk analysis
- Performance evaluation using multiple ML metrics
- Data preprocessing and feature scaling
- Data visualization using plots and charts

---

 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

 ##Dataset

The project uses the BRFSS 2015 Diabetes Health Indicators Dataset.

Dataset includes health-related attributes such as:

- High Blood Pressure
- High Cholesterol
- BMI
- Smoking habits
- Physical Activity
- General Health
- Age
- Income
- Education
- Heart Disease History
- Mental and Physical Health

Target Variable:
- `0` → Non-Diabetic
- `1` → Diabetic

---

 ##Machine Learning Model

The project uses the *Random Forest Classifier* algorithm.

 Why Random Forest?

- Handles large datasets efficiently
- Works well with classification problems
- Reduces overfitting
- Provides high accuracy
- Handles feature importance effectively

---

 ##Evaluation Metrics

The model performance is evaluated using:

- Accuracy Score
- Precision Score
- Recall Score
- F1-Score
- ROC-AUC Score
- Confusion Matrix

---

 ##Data Preprocessing

The following preprocessing steps were performed:

- Handling missing values
- Feature selection
- Data scaling using StandardScaler
- Train-test splitting
- Feature engineering
- Data visualization

---

 ##Visualizations

The project generates several plots including:

- Confusion Matrix
- ROC Curve
- Feature Importance Graph
- Correlation Heatmap
- Class Distribution Plot

---

 ##Project Structure

AI-Based-Diabetes-Prediction-System/
 app.py
 train_model.py
 requirements.txt
 model.pkl
 scaler.pkl
 feature_names.pkl
 metrics.pkl
 
 plots:
     confusion_matrix.png
     roc_curve.png
     feature_importance.png
     correlation_heatmap.png
     class_distribution.png
 
 dataset:
     diabetes_binary_5050split_health_indicators_BRFSS2015.csv
 
