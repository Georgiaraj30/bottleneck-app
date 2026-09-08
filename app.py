import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and feature baseline
model = joblib.load("bottleneck_model.pkl")
feature_means = joblib.load("feature_means.pkl")

st.title("IT System Bottlenecks on Server and Network Performance Logs")
st.write("Simulated real-time monitoring using ML")

# User-controlled sliders
cpu_user = st.slider("CPU User Usage", 0.0, 1.0, 0.3)
cpu_iowait = st.slider("CPU IO Wait", 0.0, 1.0, 0.1)
disk_io_time = st.slider("Disk IO Time", 0.0, 1.0, 0.02)

# Create full input with baseline values
input_df = pd.DataFrame([feature_means])

# Override key metrics
input_df["cpu-user"] = cpu_user
input_df["cpu-iowait"] = cpu_iowait
input_df["disk-io-time"] = disk_io_time

# Predict bottleneck risk
if st.button("Predict Bottleneck Risk"):
    prob = model.predict_proba(input_df)[0][1]   # Probability of bottleneck
    risk_percent = round(prob * 100, 2)

    # Risk interpretation
    if risk_percent < 30:
        st.success(f"✅ LOW RISK: {risk_percent}%")
    elif risk_percent < 70:
        st.warning(f"⚠️ MEDIUM RISK: {risk_percent}%")
    else:
        st.error(f"🚨 HIGH RISK: {risk_percent}%")

        # -------- ALERT & RECOMMENDATION SYSTEM --------

    st.subheader("Bottleneck Analysis")

    causes = []
    actions = []

    if cpu_user > 0.7:
        causes.append("High CPU User Usage")
        actions.append("Reduce running processes or scale CPU resources")

    if cpu_iowait > 0.3:
        causes.append("High CPU IO Wait")
        actions.append("Check disk performance or optimize I/O operations")

    if disk_io_time > 0.02:
        causes.append("High Disk IO Time")
        actions.append("Increase disk IOPS or postpone heavy disk tasks")

    if causes:
        st.error("Bottleneck Causes Detected:")
        for c in causes:
            st.write("•", c)

        st.info("🛠 Recommended Actions:")
        for a in actions:
            st.write("•", a)
    else:
        st.success(" No critical bottleneck causes identified")
