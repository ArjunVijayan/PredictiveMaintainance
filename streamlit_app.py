import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

from model import FailureTimeModel

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.style.use("ggplot")

st.sidebar.title("Machine Failure Prediction")
st.sidebar.markdown("This app enables users to predict potential machine failures using machine learning.")

options = st.sidebar.radio("Pages", options=["Model Configuration", "Estimate Failure Time", "Plot Hazard Function"])

def configure_model():

    st.markdown("### Configure Model")

    path = "/Users/arjun-14756/Desktop/survival analysis demo/"

    data = ["failure_data"]
    dat = st.selectbox("Select Data", data)

    if dat is not None:
        df = pd.read_csv(f"{path}{dat}.csv")
        st.dataframe(df.groupby("ID").first().reset_index(), height=250)

        st.info("The dataset includes machine failure data with a unique ID for each machine, a TimeToFailure column indicating when each machine fails, and additional columns for sensor data associated with the machines.")

        ids = ["ID"]
        times = ["TimeToFailure"]
        events = ["Event"]

        id_column = st.selectbox("Select ID Column :- Unique identifier for each machine/unit.", ids)
        duration_column = st.selectbox("Select Duration Column :- Time until event occurrence or end of observation.", times)
        event_column = st.selectbox("Select Event Column :- Binary indicator of event occurrence (1 for event, 0 for no event).", events)

        train = st.button("Train Hazard Model")
        model = FailureTimeModel(df, id_column=id_column
        , duration_column=duration_column, event_column=event_column)

        if train:
            with st.spinner("Please Wait.."):

                mod, score = model.train_model_()

                st.markdown("### Training Response")
                st.json({"Training Status": "Success"
                , "Concordance Score":round(score, 2)})

                st.info("In survival analysis, the concordance score (C-index) measures how well a model ranks survival times based on predicted risks. A score of 1 indicates perfect prediction, 0.5 suggests random prediction, and below 0.5 indicates poor prediction.")


        st.session_state.model = model
        st.session_state.data = df
        st.session_state.id_column = id_column
        st.session_state.duration_col = duration_column
        st.session_state.event_column = event_column

def expected_failure_times():
    st.header("Expected Failure Time")
    st.write("Functionality to predict the expected time to failure.")
    
    model = st.session_state.model
    data = st.session_state.data

    event_column = st.session_state.event_column
    id_column = st.session_state.id_column
    duration_column = st.session_state.duration_col

    data = data.groupby("ID").first().reset_index()
    dat_ = data.drop(event_column, axis=1)

    st.dataframe(dat_.drop(duration_column,  axis=1), height=250)

    individuals = list(dat_[id_column].unique())
    individuals_selected = st.multiselect("Select Individuals", individuals)
    n = len(individuals_selected)

    if n >= 1:
        dat_ = dat_[dat_[id_column].isin(individuals_selected)]
        
        st.markdown("Selected Individuals")
        st.dataframe(dat_.drop(duration_column, axis=1), height=250)

        estimate_ft = st.button("Get estimated failure time")

        if estimate_ft:

            ranked_df = model.rank_machine_failures_(dat_)        
            survival_df = model.estimate_ttmf_(dat_)

            ranked_df["Expeted_FT"] = survival_df["Expected Time"]
            tab1, tab2 = st.tabs(["Tabular View", "Graphical View"])

            with tab1:
                st.header("Expected FTime")
                st.markdown("Expected failure time refers to the estimated duration until an event.")
                ranked_df.drop("RiskScore", axis=1, inplace=True)
                st.dataframe(ranked_df, height=250)

            with tab2:

                st.header("Expected FTime")
                st.markdown("Expected failure time refers to the estimated duration until an event.")
                plt.figure(figsize=(15, 5))
                fig = sns.barplot(y="Expeted_FT", x="ID", data=ranked_df)
                fig = plt.savefig('my_plot.png')
                st.pyplot(fig)


def visualize_results():
    st.header("Visualise Results")
    st.write("Functionality to predict the expected time to failure.")
    
    model = st.session_state.model
    data = st.session_state.data

    event_column = st.session_state.event_column
    id_column = st.session_state.id_column
    duration_column = st.session_state.duration_col

    data = data.groupby("ID").first().reset_index()
    dat_ = data.drop(event_column, axis=1)

    st.dataframe(dat_.drop(duration_column, axis=1), height=250)
    
    individuals = list(dat_[id_column].unique())
    individuals_selected = st.multiselect("Select Individuals", individuals)

    n = len(individuals_selected)

    if n >= 1:
        dat_ = dat_[dat_[id_column].isin(individuals_selected)]
        
        st.markdown("Selected Individuals")
        st.dataframe(dat_.drop(duration_column, axis=1))

        survival_button = st.button("estimate hazard function")

        if survival_button:
            st.markdown("The hazard function in survival analysis quantifies the instantaneous event occurrence rate at any given time, considering survival up to that point.")
            survival_func = model.estimate_survival_function_(dat_)
            tab1, tab2 = st.tabs(["Tabular View", "Graphical View"])

            with tab1:
                st.dataframe(survival_func)

            with tab2:
                ids = survival_func["ID"]

                survival_func = survival_func.drop("ID", axis=1)
                survival_func_t = survival_func.transpose()
                
                survival_func_t.columns = ids
                
                st.line_chart(survival_func_t)

                st.info("The hazard function plot depicts the time until failure on the x-axis and the probability of failure occurring at that specific point in time on the y-axis.")

if options == "Model Configuration":
    configure_model()

elif options == "Estimate Failure Time":
    expected_failure_times()

elif options == "Plot Hazard Function":
    visualize_results()