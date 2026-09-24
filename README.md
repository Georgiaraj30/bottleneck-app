Detection of IT System Bottlenecks Using Machine Learning and MLOps

## About

IT systems can experience performance degradation when server resources are heavily utilized or when network activity increases. These performance issues can lead to slow response times, reduced system stability, and potential downtime.

This project focuses on detecting IT system bottlenecks using machine learning applied to server and network performance logs.

The project has been extended from a traditional machine learning application into an end-to-end MLOps workflow using Git, DVC, MLflow, and Apache Airflow.

The system analyzes server and network performance indicators such as:

- CPU User Usage
- CPU I/O Wait
- Disk I/O Time
- Disk I/O Activity
- Network Information
- Connection and Protocol Information

The trained machine learning models predict the probability of a system bottleneck. The Streamlit application presents the prediction as Low, Medium, or High Risk and provides possible bottleneck causes and recommended actions.


## Objectives

The main objectives of this project are:

- Detect IT system bottlenecks using machine learning.
- Combine server and network performance logs.
- Identify important system performance indicators associated with bottlenecks.
- Compare multiple machine learning models.
- Version datasets using DVC.
- Track experiments and models using MLflow.
- Automate the machine learning workflow using Apache Airflow.
- Maintain source-code version control using Git and GitHub.
- Provide an interactive prediction dashboard using Streamlit.
- Create a reproducible end-to-end MLOps workflow.

## Project Architecture


                Server Logs
                    +
                Network Logs
                    │
                    ▼
              ┌─────────────┐
              │     DVC     │
              │ Data Version│
              │   Control   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   Airflow   │
              │ Orchestration│
              └──────┬──────┘
                     │
                     ▼
             ┌────────────────┐
             │ Preprocessing  │
             │                │
             │ Data Cleaning  │
             │ Data Merging   │
             │ Feature Setup  │
             └───────┬────────┘
                     │
                     ▼
             ┌────────────────┐
             │ Model Training │
             └───────┬────────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
     Logistic     Random      XGBoost
    Regression    Forest
          │          │          │
          └──────────┼──────────┘
                     ▼
              ┌─────────────┐
              │   MLflow    │
              │ Experiment  │
              │  Tracking   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Evaluation  │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │  Streamlit  │
              │  Dashboard  │
              └─────────────┘
## Live Demo

Open the Streamlit Dashboard-(https://bottleneck-app-nbruxgom4djus7xrkbf366.streamlit.app/)
