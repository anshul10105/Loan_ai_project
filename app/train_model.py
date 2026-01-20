import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os

print("Starting training...")

# Dummy dataset (simple & safe)
data = {
    "income": [30000, 40000, 50000, 60000, 70000],
    "loan_amount": [10000, 15000, 20000, 25000, 30000],
    "credit_score": [650, 700, 720, 750, 780],
    "approved": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["income", "loan_amount", "credit_score"]]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)

with open("model/loan_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model/loan_model.pkl")
print(y.value_counts())