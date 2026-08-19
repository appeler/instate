"""Interactive web interface for Instate lookups and predictions."""

import pandas as pd

import instate
import streamlit as st

FUNCTIONS = {
    "Electoral-roll state distribution": instate.get_state_distribution,
    "BiLSTM state prediction": instate.predict_state,
}


def download_file(frame: pd.DataFrame) -> None:
    """Offer a DataFrame as a CSV download."""
    st.download_button(
        "Download results",
        frame.to_csv(index=False),
        file_name="results.csv",
        mime="text/csv",
    )


def app() -> None:
    """Render the Instate Streamlit interface."""
    st.title("Instate: estimate state patterns from last names")
    st.write(
        "Instate returns aggregate patterns from 2017 Indian electoral rolls "
        "or a bundled character-level BiLSTM. These estimates do not verify an "
        "individual's residence."
    )
    st.write("[GitHub](https://github.com/appeler/instate)")

    selected = st.sidebar.selectbox("Method", list(FUNCTIONS))
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is None:
        st.stop()

    frame = pd.read_csv(uploaded_file)
    name_column = st.selectbox("Column containing last names", frame.columns)

    if st.button("Run"):
        result = FUNCTIONS[selected](frame, name_column=name_column)
        st.dataframe(result, use_container_width=True)
        download_file(result)


if __name__ == "__main__":
    app()
