import streamlit as st
import pandas as pd
import instate
from instate import last_state, pred_last_state
import base64


# Define your sidebar options
sidebar_options = {
    'Append Indian Electoral Roll Data': last_state,
    'Predict': pred_last_state
}

def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="results.csv">Download results</a>'
    st.markdown(href, unsafe_allow_html=True)

def app():
    # Set app title
    st.title("instate: predict the state of residence from last name")

    # Generic info.
    st.write('Using the Indian electoral rolls data (2017), we provide a Python package that takes the last name of a person and gives its distribution across states.')
    st.write('[Github](https://github.com/appeler/instate)')

    # Set up the sidebar
    st.sidebar.title('Select Function')
    selected_function = st.sidebar.selectbox('', list(sidebar_options.keys()))

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data loaded successfully!")
    else:
        st.stop()

    lname_col = st.selectbox("Select column with last name", df.columns)
    function = sidebar_options[selected_function]
    if st.button('Run'):
        transformed_df = function(df, lastnamecol=lname_col)
        st.dataframe(transformed_df)
        download_file(transformed_df)



# Run the app
if __name__ == "__main__":
    app()
