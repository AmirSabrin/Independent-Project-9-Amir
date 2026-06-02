import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# =====================================================
# LOAD MODELS
# =====================================================

penguin_model = joblib.load('opt_SVM_model.joblib')

iris_model = joblib.load('iris_model.pkl')

# diabetes_model = joblib.load('diabetes_model.pkl')

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    selected = option_menu(
        "Multiple Classifications System",
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
# PENGUIN SECTION
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
            "Bill Length",
            format="%.1f",
            value=30.0,
            min_value=30.0,
            max_value=60.0,
            step=0.1
        )

    with col3:
        bill_depth_mm = st.number_input(
            "Bill Depth",
            format="%.1f",
            value=12.0,
            min_value=12.0,
            max_value=20.0,
            step=0.1
        )

    with col4:
        flipper_length_mm = st.number_input(
            "Flipper Length",
            format="%d",
            value=150,
            min_value=150,
            max_value=300,
            step=1
        )

    with col5:
        body_mass = st.number_input(
            "Body Mass",
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
        ).tolist()

        st.success(
            f"Predicted Species : {prediction[0]}"
        )

# =====================================================
# IRIS SECTION
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
# DIABETES SECTION
# =====================================================

elif selected == 'Diabetes prediction':

    st.title("Diabetes Prediction")

    st.warning(
        "Diabetes model has not been created yet. "
        "Create diabetes_model.pkl first."
    )