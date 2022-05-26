import os
import streamlit as st
import numpy as np
from PIL import  Image

# Custom imports 
from multipage import MultiPage
from pages import data_upload, machine_learning, metadata, data_visualize, redundant, Tayo,  tise# import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page

st.set_page_config(
    page_title='Macroeconomic',
    # layout="wide"
)

display = Image.open('Logo.jpeg')
display = np.array(display)
col1, col2 = st.columns(2)
col1.image(display, width = 800)
st.title("Economic Modelling Explainer Dashboard")

# Add all your application here
app.add_page("Upload Data", data_upload.app)
#app.add_page("Change Metadata", metadata.app)
app.add_page("Variable Selection", machine_learning.app)
#app.add_page("Data Analysis",data_visualize.app)
#app.add_page("Y-Parameter Optimization",redundant.app)
app.add_page("Run Model",Tayo.app)
app.add_page("Model Output",tise.app)


# The main app
app.run()
