import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("asl_landmarks.csv")
X = data.drop("label", axis=1)  # Features (landmarks)
y = data["label"]  # Labels (A-Z, space, del, nothing)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
with open("asl_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'asl_model.pkl'")