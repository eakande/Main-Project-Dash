import streamlit.components.v1 as components
import streamlit as st

def app():
    """
    Set appearance to wide mode.
    """

    dashboardurl = 'http://127.0.0.1:8050/'
    st.components.v1.iframe(dashboardurl, width=1100, height=1100, scrolling=True)
