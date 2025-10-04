import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 🎨 Page Config
st.set_page_config(page_title="Customer Segmentation", page_icon="🌸", layout="wide")

# 🌸 Custom pastel CSS
st.markdown("""
    <style>
        body {
            background-color: #fafafa;
        }
        .stSidebar {
            background-color: #f9f3ff;
        }
        h1, h2, h3 {
            color: #6a4c93;
        }
    </style>
""", unsafe_allow_html=True)

# 🏷️ Title
st.title("🌸 Customer Segmentation with DBSCAN")
st.write("Cool pastel tones, interactive clustering, and sidebar controls!")

# 📂 File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Sidebar options
    st.sidebar.header("⚙️ DBSCAN Parameters")
    eps_val = st.sidebar.slider("Epsilon (eps)", 0.1, 10.0, 0.5, 0.1)
    min_samples_val = st.sidebar.slider("Min Samples", 2, 20, 5, 1)

    # Feature selection
    st.sidebar.header("📌 Feature Selection")
    features = st.sidebar.multiselect("Select features for clustering", df.columns, default=df.columns[:2])

    if len(features) >= 2:
        # Preprocessing
        X = df[features].values
        X_scaled = StandardScaler().fit_transform(X)

        # DBSCAN clustering
        db = DBSCAN(eps=eps_val, min_samples=min_samples_val).fit(X_scaled)
        labels = db.labels_

        # Add results to dataframe
        df['Cluster'] = labels

        st.subheader("🔎 Clustered Data")
        st.dataframe(df.head())

        # 🎨 Plot with pastel colors
        st.subheader("🎨 Cluster Visualization")
        plt.figure(figsize=(8,6))
        unique_labels = set(labels)
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 0.3]  # noise as grey
            class_member_mask = (labels == k)
            xy = X[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=f"Cluster {k}", edgecolors="k", s=80)

        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title("Customer Segments (DBSCAN)")
        plt.legend()
        st.pyplot(plt)

    else:
        st.warning("Please select at least 2 features for clustering.")
else:
    st.info("⬆️ Upload a CSV file to begin.")

