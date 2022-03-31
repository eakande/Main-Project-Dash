# Import necessary libraries
import json
import joblib
from flask import Flask
import pandas as pd
import streamlit as st

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from explainerdashboard import ExplainerDashboard, ClassifierExplainer,  InlineExplainer, RegressionExplainer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from explainerdashboard.custom import*
from dash_bootstrap_components.themes import FLATLY, CYBORG,PULSE,DARKLY

import dash_bootstrap_components as dbc
import streamlit.components.v1 as components

# Custom classes
from .utils import isNumerical
import os


def app():
    """This application helps in running machine learning models without having to write explicit code
    by the user. It runs some basic models and let's the user select the X and y variables.
    """

    # Load the data
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/main_data.csv')

        # Create the model parameters dictionary
        params = {}

        # Use two column technique
        col1, col2 = st.columns(2)

        # Design column 1
        y_var = col1.radio("Select the variable to be predicted (y)", options=data.columns)

        # Design column 2
        X_var = col2.multiselect("Select the variables to be used for prediction (X)", options=data.columns)

        # Check if len of x is not zero
        if len(X_var) == 0:
            st.error("You have to put in some X variable and it cannot be left empty.")

        # Check if y not in X
        if y_var in X_var:
            st.error("Warning! Y variable cannot be present in your X-variable.")

        # Option to select predition type
        pred_type = st.radio("Select the type of process you want to run.",
                             options=["Regression", "Classification"],
                             help="Write about reg and classification")

        # Add to model parameters
        params = {
            'X': X_var,
            'y': y_var,
            'pred_type': pred_type,
        }

        # if st.button("Run Models"):

        st.write(f"**Variable to be predicted:** {y_var}")
        st.write(f"**Variable to be used for prediction:** {X_var}")

        # Divide the data into test and train set
        X = data[X_var]
        y = data[y_var]

        # Perform data imputation
        # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")

        # Perform encoding
        X = pd.get_dummies(X)

        # Check if y needs to be encoded
        if not isNumerical(y):
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Print all the classes
            st.write("The classes and the class allotted to them is the following:-")
            classes = list(le.classes_)
            for i in range(len(classes)):
                st.write(f"{classes[i]} --> {i}")

        # Perform train test splits
        st.markdown("#### Train Test Splitting")
        size = st.slider("Percentage of value division",
                         min_value=0.1,
                         max_value=0.9,
                         step=0.1,
                         value=0.8,
                         help="This is the value which will be used to divide the data for training and testing. Default = 80%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        st.write("Number of training samples:", X_train.shape[0])
        st.write("Number of testing samples:", X_test.shape[0])



        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        explainer = RegressionExplainer(model, X_test, y_test)

        #ExplainerDashboard(explainer,  bootstrap=dbc.themes.CYBORG).run()

        class CustomModelTab(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Selected Drivers")
                self.importance = ImportancesComposite(explainer,
                                                       title='Impact',
                                                       hide_importances=False)
                self.register_components()

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            self.importance.layout(),
                            html.H3(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                                    f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                                    f" and {self.explainer.columns_ranked_by_shap()[2]}.")
                        ])
                    ])
                ])

        class CustomModelTab1(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Model Performance")
                self.Reg_summary = RegressionModelStatsComposite(explainer,
                                                                 title='Impact',
                                                                 hide_predsvsactual=False, hide_residuals=False,
                                                                 hide_regvscol=False)
                self.register_components()

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            self.Reg_summary.layout(),

                        ])
                    ])
                ])

        class CustomPredictionsTab(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Model Predictions")

                self.prediction = IndividualPredictionsComposite(explainer,
                                                                 hide_predindexselector=False, hide_predictionsummary=False,
                                                                 hide_contributiongraph=False, hide_pdp=False,
                                                                 hide_contributiontable=False)
                self.register_components()

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Individual Prediction:"),
                            self.prediction.layout()
                        ])

                    ])
                ])

        class CustomPredictionsTab2(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="What if Scenarios")

                self.what_if = WhatIfComposite(explainer,
                                               hide_whatifindexselector=False, hide_inputeditor=False,
                                               hide_whatifcontribution=False, hide_whatifpdp=False)
                self.register_components()

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Individual Prediction:"),
                            self.what_if.layout()
                        ])

                    ])
                ])

        class CustomPredictionsTab3(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="SHAP Dependencies")

                self.shap_depend = ShapDependenceComposite(explainer,
                                                           hide_shapsummary=False, hide_shapdependence=False)
                self.register_components()

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("SHAP Dependencies:"),
                            self.shap_depend.layout()
                        ])

                    ])
                ])

        class CustomPredictionsTab4(ExplainerComponent):
            def __init__(self, explainer):
                super().__init__(explainer, title="Interacting Features")

                self.interaction = ShapInteractionsComposite(explainer,
                                                             hide_interactionsummary=False,
                                                             hide_interactiondependence=False)
                self.register_components()

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Interacting Features:"),
                            self.interaction.layout()
                        ])

                    ])
                ])



        ExplainerDashboard(explainer, [CustomModelTab, CustomModelTab1, CustomPredictionsTab,
                                            CustomPredictionsTab2, CustomPredictionsTab3, CustomPredictionsTab4],
                                title='Macroeconomic Indicator Prediction for Nigeria', header_hide_selector=False,
                                bootstrap=CYBORG).run()


