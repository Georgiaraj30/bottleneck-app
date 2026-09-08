Detection of IT System Bottlenecks Using Machine Learning on Server and Network Performance Logs

About

IT systems often slow down when server load increases or when there is heavy network traffic. 
These slowdowns, known as bottlenecks, can negatively affect system performance, user experience, and overall stability. 
If bottlenecks are not detected, they may lead to system failures and downtime. 
This project focuses on the detection of IT system bottlenecks using machine learning techniques applied to server and network performance logs.

- CPU User Usage
- CPU I/O Wait
- Disk I/O Time

The model predicts the bottleneck probability and classifies it as Low, Medium, or High Risk. It also provides possible causes and recommended actions.

Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Google Colab

Files
ML_PROJECT.ipynb       # ML code and results
app.py                 # Streamlit application
bottleneck_model.pkl   # Trained model
feature_means.pkl      # Feature baseline values
requirements.txt       # Dependencies
