import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import datetime

# -----------------------------
# Step 1: Create/Load Expense Data
# -----------------------------
# Sample data (Date, Category, Amount)
data = {
    "Date": ["2026-03-01", "2026-03-03", "2026-03-05", "2026-03-07", "2026-03-10"],
    "Category": ["Food", "Transport", "Food", "Entertainment", "Food"],
    "Amount": [20, 15, 30, 50, 25]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.dayofyear  # convert date to day number for prediction

# -----------------------------
# Step 2: Visualize Expenses
# -----------------------------
plt.figure(figsize=(8,5))
sns.barplot(x='Date', y='Amount', data=df)
plt.title("Daily Expenses")
plt.xlabel("Date")
plt.ylabel("Amount ($)")
plt.show()

# -----------------------------
# Step 3: Predict Future Expenses
# -----------------------------
# Features and target
X = df[['Day']]  # predictor
y = df['Amount']  # target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Predict expense for a future date
future_date = datetime.datetime(2026, 3, 15)
future_day = pd.DataFrame([[future_date.timetuple().tm_yday]], columns=['Day'])
predicted_expense = model.predict(future_day)[0]

print(f"Predicted expense on {future_date.date()}: ${predicted_expense:.2f}")