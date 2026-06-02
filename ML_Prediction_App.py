import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# =====================================================
# LOAD MODELS
# =====================================================

penguin_model = joblib.load('opt_SVM_model.joblib')
iris_model = joblib.load('iris_model.pkl')
diabetes_model = joblib.load('diabetes_model.pkl')

# =====================================================
# SIDEBAR MENU
# =====================================================

with st.sidebar:

    selected = option_menu(
        "Multiple Classification System",
        [
            'Penguin species prediction',
            'Iris species prediction',
            'Diabetes prediction'
        ],
        menu_icon='species-fill',
        icons=['emoji-grin', 'flower1', 'person'],
        default_index=0
    )

# =====================================================
# PENGUIN PREDICTION
# =====================================================

if selected == 'Penguin species prediction':

    st.title("Penguin Species Prediction Using SVM")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        island = st.selectbox(
            "Island Selection",
            ["Biscoe", "Dream", "Torgersen"]
        )

    with col2:
        bill_length_mm = st.number_input(
            "Bill Length (mm)",
            format="%.1f",
            value=30.0,
            min_value=30.0,
            max_value=60.0,
            step=0.1
        )

    with col3:
        bill_depth_mm = st.number_input(
            "Bill Depth (mm)",
            format="%.1f",
            value=12.0,
            min_value=12.0,
            max_value=20.0,
            step=0.1
        )

    with col4:
        flipper_length_mm = st.number_input(
            "Flipper Length (mm)",
            format="%d",
            value=150,
            min_value=150,
            max_value=300,
            step=1
        )

    with col5:
        body_mass = st.number_input(
            "Body Mass (g)",
            format="%d",
            value=3000,
            min_value=3000,
            max_value=6000,
            step=1
        )

    with col6:
        sex = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

    prediction_btn = st.button("Predict Penguin")

    if prediction_btn:

        keys = [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex"
        ]

        user_input = [
            island,
            bill_length_mm,
            bill_depth_mm,
            flipper_length_mm,
            body_mass,
            sex
        ]

        input_dict = dict(zip(keys, user_input))

        prediction = penguin_model.predict(
            pd.DataFrame(input_dict, index=[0])
        )

        st.success(
            f"Predicted Species : {prediction[0]}"
        )

# =====================================================
# IRIS PREDICTION
# =====================================================

elif selected == 'Iris species prediction':

    st.title("Iris Species Prediction Using SVM")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            value=5.1
        )

    with col2:
        sepal_width = st.number_input(
            "Sepal Width (cm)",
            value=3.5
        )

    with col3:
        petal_length = st.number_input(
            "Petal Length (cm)",
            value=1.4
        )

    with col4:
        petal_width = st.number_input(
            "Petal Width (cm)",
            value=0.2
        )

    iris_btn = st.button("Predict Iris")

    if iris_btn:

        iris_input = pd.DataFrame({

            'SepalLengthCm': [sepal_length],
            'SepalWidthCm': [sepal_width],
            'PetalLengthCm': [petal_length],
            'PetalWidthCm': [petal_width]

        })

        prediction = iris_model.predict(
            iris_input
        )

        st.success(
            f"Predicted Species : {prediction[0]}"
        )

# =====================================================
# DIABETES PREDICTION
# =====================================================

elif selected == 'Diabetes prediction':

    st.title("Diabetes Prediction Using SVM")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            value=1
        )

    with col2:
        glucose = st.number_input(
            "Glucose",
            min_value=0,
            value=100
        )

    with col3:
        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            value=70
        )

    with col4:
        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            value=20
        )

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        insulin = st.number_input(
            "Insulin",
            min_value=0,
            value=80
        )

    with col6:
        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0
        )

    with col7:
        pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.5
        )

    with col8:
        age = st.number_input(
            "Age",
            min_value=1,
            value=25
        )

    diabetes_btn = st.button("Predict Diabetes")

    if diabetes_btn:

        diabetes_input = pd.DataFrame({

            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [pedigree],
            'Age': [age]

        })

        prediction = diabetes_model.predict(
            diabetes_input
        )

        if prediction[0] == 1:

            st.error(
                "Patient is Diabetic"
            )

        else:

            st.success(
                "Patient is Non-Diabetic"
            )