import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy dataset
data = {
    "units": [100, 200, 300, 400, 500],
    "bill": [500, 1000, 1500, 2000, 2500]
}
df = pd.DataFrame(data)

# Features and target
X = df[['units']]
y = df['bill']

# Train model on ALL data (no train/test split)
model = LinearRegression()
model.fit(X, y)

# Example prediction
new_units = np.array([[350]])  # units consumed
predicted_bill = model.predict(new_units)[0]
print(f"Predicted Electricity Bill: ₹{predicted_bill:.2f}")
