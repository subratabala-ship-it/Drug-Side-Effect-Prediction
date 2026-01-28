import streamlit as st

st.set_page_config(
    page_title="Drug Side Effect Predictor",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Drug Side Effect Prediction System")
st.markdown(
    """
    ### Personalized Patient Care using Machine Learning  
    Predict **possible drug side effects** based on:
    - Patient age group  
    - Disease condition  

    👉 Use the **sidebar** to navigate through the app.
    """
)
