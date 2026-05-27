import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image

#Page Config 
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide"
)

# Load Model 
@st.cache_resource
def load_model():
    model         = joblib.load("model.pkl")
    scaler        = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# Header 
st.title("AI-Based Diabetes Prediction System")
st.markdown("Enter patient health details to predict diabetes risk.")
st.markdown("---")
if not model_loaded:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()

# Tabs 
tab1, tab2, tab3 = st.tabs(["Predict",  "Model Performance", "About Dataset"])

# 1.PREDICTION
with tab1:
    st.subheader("Patient Health Information")
    st.markdown("Fill in the patient details below and click **Predict**.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**General Health**")
        age = st.selectbox(
            "Age Group",
            options=list(range(1, 14)),
            format_func=lambda x: {
                1:"18–24", 2:"25–29", 3:"30–34", 4:"35–39",
                5:"40–44", 6:"45–49", 7:"50–54", 8:"55–59",
                9:"60–64", 10:"65–69", 11:"70–74", 12:"75–79", 13:"80+"
            }[x],
            index=6
        )
        st.markdown("BMI Calculator")
        with st.expander("Click to calculate your BMI"):
            height = st.number_input("Enter Height(in cm)",min_value=100.0, max_value=250.0,value =165.0)
            weight = st.number_input("Enter Weight(in kg)",min_value=30.0, max_value=200.0, value=65.0)
            if st.button("Calculate BMI"):
                bmi_calc= weight / ((height/100) **2)
                st.success(f"Your BMI is: **{bmi_calc:.2f}**")
                
                #BMI Category
                if bmi_calc <18.5:
                    st.info("Category: Underweight")
                elif 18.5 <= bmi_calc < 25:
                    st.success("Category: Normal weight")
                elif 25 <= bmi_calc < 30:
                    st.warning("Category: Overweight")
                else:
                    st.error("Category: Obese")
                
        
        bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=30.0, step=0.1,
                              help="Body Mass Index (normal: 18.5–24.9)")
        gen_hlth = st.selectbox(
            "General Health (self-rated)",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1:"Excellent", 2:"Very Good", 3:"Good", 4:"Fair", 5:"Poor"}[x],
            index=2
        )
        sex = st.radio("Sex", options=[0, 1],
                       format_func=lambda x: "Female" if x == 0 else "Male",
                       horizontal=True)

    with col2:
        st.markdown("**Medical History**")
        high_bp    = st.radio("High Blood Pressure?",   [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        high_chol  = st.radio("High Cholesterol?",      [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        chol_check = st.radio("Cholesterol Check (last 5 yrs)?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        stroke     = st.radio("History of Stroke?",     [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        heart_dis  = st.radio("Heart Disease / Attack?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        diff_walk  = st.radio("Difficulty Walking?",    [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

    with col3:
        st.markdown("**Lifestyle**")
        phys_act   = st.radio("Physical Activity (last 30 days)?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        smoker     = st.radio("Smoked cigarettes in lifetime?",  [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        fruits     = st.radio(" Frequent Fruits consumption ?",   [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        veggies    = st.radio("Including Vegetables in daily routine ?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        hvy_alc    = st.radio("Heavy Alcohol Consumption?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

        st.markdown("**Healthcare Access**")
        any_healthcare = st.radio("Has Healthcare Coverage?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        no_doc_cost    = st.radio("Couldn't afford doctor visit?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ment_hlth = st.slider("Poor Mental Health Days (last 30 days)", 0, 30, 3)
    with col_b:
        phys_hlth = st.slider("Poor Physical Health Days (last 30 days)", 0, 30, 3)
    with col_c:
        education = st.selectbox(
            "Education Level",
            options=[1, 2, 3, 4, 5, 6],
            format_func=lambda x: {
                1:"Never attended school", 2:"Elementary",
                3:"High school dropout", 4:"High school graduate",
                5:"Undergraduate dropout", 6:"College graduate"
            }[x],
            index=4
        )
        income = st.selectbox(
            "Income Level",
            options=list(range(1, 9)),
            format_func=lambda x: {
                1:"< $10K", 2:"$10K–$15K", 3:"$15K–$20K", 4:"$20K–$25K",
                5:"$25K–$35K", 6:"$35K–$50K", 7:"$50K–$75K", 8:"> $75K"
            }[x],
            index=5
        )

    st.markdown("---")
    predict_btn = st.button("Predict Diabetes Risk", use_container_width=True, type="primary")

    if predict_btn:
        input_data = {
            "HighBP": high_bp, "HighChol": high_chol, "CholCheck": chol_check,
            "BMI": bmi, "Smoker": smoker, "Stroke": stroke,
            "HeartDiseaseorAttack": heart_dis, "PhysActivity": phys_act,
            "Fruits": fruits, "Veggies": veggies, "HvyAlcoholConsump": hvy_alc,
            "AnyHealthcare": any_healthcare, "NoDocbcCost": no_doc_cost,
            "GenHlth": gen_hlth, "MentHlth": ment_hlth, "PhysHlth": phys_hlth,
            "DiffWalk": diff_walk, "Sex": sex, "Age": age,
            "Education": education, "Income": income
        }

        input_df  = pd.DataFrame([input_data])[feature_names]
        scaled    = scaler.transform(input_df)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]
        risk_pct    = probability * 100

        st.markdown("###Prediction Result")
        r1, r2 = st.columns(2)

        with r1:
            if prediction == 1:
                st.error(f"### Diabetic / Pre-Diabetic")
                st.markdown(f"**Risk Probability: `{risk_pct:.1f}%`**")
                st.progress(probability)
                st.warning("High risk detected. Please consult a healthcare professional.")
            else:
                st.success(f"###Non-Diabetic")
                st.markdown(f"**Risk Probability: `{risk_pct:.1f}%`**")
                st.progress(probability)
                st.info("Low risk. Maintain a healthy lifestyle to stay protected.")

        with r2:
            st.markdown("**Top Risk Factors in Your Input:**")
            risk_factors = {
                "High Blood Pressure": high_bp,
                "High Cholesterol": high_chol,
                "BMI > 30 (Obese)": int(bmi >= 30),
                "No Physical Activity": int(phys_act == 0),
                "History of Stroke": stroke,
                "Heart Disease": heart_dis,
                "Poor General Health": int(gen_hlth >= 4),
                "Difficulty Walking": diff_walk,
                "Heavy Alcohol Use": hvy_alc,
                "Smoker": smoker,
            }
            flagged = [k for k, v in risk_factors.items() if v == 1]
            if flagged:
                for f in flagged:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- No major risk factors flagged")

# 2.MODEL PERFORMANCE

with tab2:
    st.subheader("Model Evaluation & Visualizations")

    # Metrics summary
    st.markdown("#### Performance Metrics (on 20% test set)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",  "75%")
    m2.metric("Precision", "75%")
    m3.metric("Recall",    "75%")
    m4.metric("F1-Score",  "75%")
    m5.metric("ROC-AUC",   "0.83")

    st.markdown("---")

    #Saved plots
    plots = {
        "Confusion Matrix":        "plots/confusion_matrix.png",
        "ROC-AUC Curve":           "plots/roc_curve.png",
        "Feature Importance":      "plots/feature_importance.png",
        "Correlation Heatmap":     "plots/correlation_heatmap.png",
        "class_distribution":        "plots/class_distribution.png"
    }
    available = {k: v for k, v in plots.items() if os.path.exists(v)}
    if not available:
        st.info("Run `python train_model.py` first to generate the plots.")
    else:
        cols = st.columns(2)
        for i, (title, path) in enumerate(available.items()):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                img = Image.open(path)
                st.image(img, use_container_width=True)

#3.ABOUT
with tab3:
    st.subheader("About the Dataset")
    st.markdown("""
    **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015  
    **Kaggle:** [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

    | Property | Value |
    |---|---|
    | Total Rows | 70,692 |
    | Features | 21 |
    | Target | `Diabetes_binary` (0 = No Diabetes, 1 = Diabetic/Pre-diabetic) |
    | Class Balance | 50% – 50% (perfectly balanced) |
    | Missing Values | None |

    #### Feature Descriptions
    | Feature | Description |
    |---|---|
    | HighBP | High blood pressure (0=No, 1=Yes) |
    | HighChol | High cholesterol (0=No, 1=Yes) |
    | CholCheck | Cholesterol check in last 5 years |
    | BMI | Body Mass Index |
    | Smoker | Smoked 100+ cigarettes in lifetime |
    | Stroke | History of stroke |
    | HeartDiseaseorAttack | Coronary heart disease or heart attack |
    | PhysActivity | Physical activity in past 30 days |
    | Fruits | Consume fruit daily |
    | Veggies | Consume vegetables daily |
    | HvyAlcoholConsump | Heavy alcohol consumption |
    | AnyHealthcare | Has any healthcare coverage |
    | NoDocbcCost | Could not see doctor due to cost |
    | GenHlth | General health rating (1=Excellent to 5=Poor) |
    | MentHlth | Days of poor mental health (0–30) |
    | PhysHlth | Days of poor physical health (0–30) |
    | DiffWalk | Difficulty walking or climbing stairs |
    | Sex | 0=Female, 1=Male |
    | Age | Age group (1=18–24 ... 13=80+) |
    | Education | Education level (1–6) |
    | Income | Income level (1–8) |

    #### Model
    - **Algorithm:** Random Forest Classifier (100 trees, max depth 10)
    - **Preprocessing:** StandardScaler on all features
    - **Split:** 80% train / 20% test (stratified)
    """)

# Footer 
st.markdown("---")
st.markdown(
    "<center><small>AI-Based Diabetes Prediction System | Minor Project | Powered by Scikit-learn & Streamlit</small></center>",
    unsafe_allow_html=True
)
