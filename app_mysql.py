import streamlit as st
import pandas as pd
import mysql.connector
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Database Connection
# -----------------------------
def get_data():
    # Connect to MySQL (update with your own credentials)
    conn = mysql.connector.connect(
        host="localhost",       # XAMPP default
        user="root",            # default user
        password="",            # leave empty if no password
        database="electricity_bill" # replace with your DB name
    )
    query = "SELECT * FROM your_table;"  # replace with your table name
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 MySQL Data Analysis & Visualization")

# Load data
df = get_data()
st.write("### Raw Data Preview")
st.dataframe(df)

# -----------------------------
# Basic Analysis
# -----------------------------
st.write("### Summary Statistics")
st.write(df.describe())

# -----------------------------
# Visualization Options
# -----------------------------
st.write("### Visualizations")

# Select column for visualization
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
selected_col = st.selectbox("Select a numeric column to visualize:", numeric_cols)

# Histogram
fig, ax = plt.subplots()
sns.histplot(df[selected_col], kde=True, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
