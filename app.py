import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

def get_binary_file_downloaded_html(df):
    csv = df.to_csv(index=False)
    b64  = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV File</a>'
    return href

st.title("Heart Disease Risk Analyzer")
tab1,tab2,tab3= st.tabs(["Predict","Bulk Predict","Model Information"]) # Tabs for different functionalities

with tab1:
    age = st.number_input("Age(years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex",["Male","Female"])
    chest_pain = st.selectbox("Chest Pain Type",["Typical Angina","Atypical Angina","Non-Anginal Pain","Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/d", "> 120 mg/d"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical inputs to numerical values

    sex = 0 if sex == "Male" else 1
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == "<= 120 mg/d" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # create a DataFrame for the input
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })


    # Load the model
    algonames = ['Desicision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'GridRandom']
    modelnames = ['DT.pkl', 'LogisticR.pkl', 'RDT.pkl', 'SVM.pkl', 'GRF.pkl']

    predictions = []
    def predict_heart_disease(data):
        for modelname in modelnames:
            model = pickle.load(open(modelname, 'rb'))
            prediction = model.predict(data)
            predictions.append(prediction)
        return predictions
    
    # Create a button to make predictions
    if st.button("Submit"):
        st.subheader("Results")
        st.markdown("--------------------------------------")
        
        result = predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0] == 0:
                st.success("No Heart Disease Risk Detected")
            else:
                st.error("Heart Disease Risk Detected")
            st.markdown("--------------------------------------")

with tab2:
    st.title("Upload CSV File")

    st.subheader("Instructions to note before uploading the file:")
    st.info("""
            1. No NaN values allowed.
            2. Total 11 features in the order('Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
            'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope').\n
            3. Check the spellings of the features names.\n
            4. Feature values conventions:\n
                - Age: age of the patient [years]\n
                -Sex: sex of the patient [0:Male, 1:Female]\n
                - ChestPainType: chest pain type [3: Typical Angina, 0:Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]\n
                - RestingBP: resting blood pressure [mm Hg]\n
                - Cholesterol: serum cholesterol [mg/dl]\n
                - FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]\n
                - RestingECG: resting electrocardiographic results [0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy]\n
                - MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]\n
                - ExerciseAngina: exercise induced angina [1:yes, 0: No]\n
                - Oldpeak: oldpeak = ST[Numeric value measured in depression]\n
                - ST_Slope: slope of the peak exercise ST segment [0: Upsloping, 1: Flat, 2: Downsloping]\n
                """)
    
    #create a file uploader in the sidebar

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                            'Oldpeak', 'ST_Slope']

        if set(expected_columns).issubset(input_data.columns):
            # Extract only the features for prediction
            features = input_data[expected_columns]

            # Load all models
            models = {
                'Prediction_LR': pickle.load(open('LogisticR.pkl', 'rb')),
                'Prediction_DT': pickle.load(open('DT.pkl', 'rb')),
                'Prediction_RF': pickle.load(open('RDT.pkl', 'rb')),
                'Prediction_SVM': pickle.load(open('SVM.pkl', 'rb')),
                'Prediction_GridRF': pickle.load(open('GRF.pkl', 'rb'))
            }

            # Apply each model to the feature subset and store predictions
            for col, model in models.items():
                input_data[col] = model.predict(features)

            st.subheader("Predictions:")
            st.write(input_data)

            # Download link
            st.markdown(get_binary_file_downloaded_html(input_data), unsafe_allow_html=True)

        else:
            st.warning("The uploaded file does not contain the required columns.")
    else:
        st.info("Upload a CSV file to see the predictions.")

with tab3:
    import plotly.express as px
    
    # Proper dictionary with model name as key and accuracy as value
    data = {
        'Decision Tree': 80.97,
        'Logistic Regression': 85.86,
        'SVM': 84.22,
        'Random Forest': 84.78,
        'Grid Random Forest': 87.5
    }

    Models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(Models, Accuracies)), columns=['Model', 'Accuracy'])

    fig = px.bar(df, x='Model', y='Accuracy', title="Model Accuracies", text='Accuracy')
    st.plotly_chart(fig)

    