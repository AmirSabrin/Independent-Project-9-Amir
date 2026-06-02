import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu



# print(working_dir)

# load the saved model

model = joblib.load('opt_SVM_model.joblib')
# print(model)

# sidebar for navigation

with st.sidebar:
    selected = option_menu("Multiple Classifications System",
                           ['Penguin species prediction',
                            'Iris species prediction',
                            'Diabetes prediction',
                            ],
                           menu_icon='species-fill',
                           icons=['emoji-grin', 'apple', 'person'],
                           default_index=0)

if selected == 'Penguin species prediction':
    # page title
    st.title("Penguin species prediction using SVM")

    # get input
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        island = st.selectbox("Island selection", ["Biscoe", "Dream", "Torgersen"])

    with col2:
        bill_length_mm = st.number_input("Enter bill length", format="%.1f", value=30.0, min_value=30.0, max_value=60.0,
                                         step=0.1)

    with col3:
        bill_depth_mm = st.number_input("Enter bill depth", format="%.1f", value=12.0, min_value=12.0, max_value=20.0,
                                        step=0.1)

    with col4:
        flipper_length_mm = st.number_input("Enter flipper length", format="%d", value=150, min_value=150,
                                            max_value=300,
                                            step=1)

    with col5:
        body_mass = st.number_input("Enter body mass", format="%d", value=3000, min_value=3000,
                                    max_value=6000,
                                    step=1)

    with col6:
        sex = st.selectbox("Gender", ["Male", "Female"])


#code for classification

spec_classification = ''

keys = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]

#combine keys and values into a dictionary ["island": 'Dream']

prediction_btn = st.button("Prediction")

if prediction_btn:
    user_input  = [island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass, sex]

    input_dict = dict(zip(keys, user_input))
    spec_classification = model.predict(pd.DataFrame(input_dict,index=[0])).tolist()
    #st.write(spec_classification)
st.success(spec_classification)
