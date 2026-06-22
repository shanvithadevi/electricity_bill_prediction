import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Dummy dataset with extra factors
data = {
    "units": [100, 200, 300, 400, 500, 600],
    "season": [0, 0, 1, 1, 0, 1],   # 0 = winter, 1 = summer
    "appliances": [5, 7, 10, 12, 8, 15],
    "daily_usage": [3, 4, 6, 7, 5, 8],
    "bill": [500, 1000, 1800, 2500, 1600, 3200]
}
df = pd.DataFrame(data)

# Features and target
X = df[['units', 'season', 'appliances', 'daily_usage']]
y = df['bill']

# Train model on ALL data
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("Electricity Bill Prediction⚡")

units = st.number_input("Enter units consumed (kWh):", min_value=50, max_value=1000, step=10)
season = st.selectbox("Season:", ["Winter", "Summer"])
appliances = st.slider("Number of appliances:", 1, 20, 5)
daily_usage = st.slider("Average daily usage (hours):", 1, 24, 6)

season_val = 1 if season == "Summer" else 0

if st.button("Predict Bill"):
    input_data = np.array([[units, season_val, appliances, daily_usage]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Electricity Bill: ₹{prediction:.2f}")

    # 🎨 Visualization with pastel colors
    st.write("### Data Visualization")

    # Histogram of units
    fig, ax = plt.subplots()
    sns.histplot(df['units'], kde=True, color="lightpink", ax=ax)
    ax.set_title("Units Consumed Distribution", fontsize=14, color="darkblue")
    st.pyplot(fig)

    # Scatter plot: units vs bill
    fig, ax = plt.subplots()
    sns.scatterplot(x="units", y="bill", data=df, hue="season", palette="pastel", s=100, ax=ax)
    ax.set_title("Units vs Bill (Pastel Theme)", fontsize=14, color="darkgreen")
    st.pyplot(fig)

    # Correlation heatmap
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="Pastel1", ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, color="purple")
    st.pyplot(fig)

# Add custom pastel background (optional)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fdf6f0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
