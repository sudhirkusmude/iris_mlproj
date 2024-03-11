# ml_script.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset (example: using the Iris dataset)
def load_data():
    iris_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    iris_data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    return iris_data

# Preprocess data
def preprocess_data(data):
    X = data.drop("class", axis=1)
    y = data["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Save model
def save_model(model, model_filename="model.pkl"):
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

# Main function
def main():
    # Step 1: Load data
    data = load_data()

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model accuracy: {accuracy}")

    # Step 5: Save model
    save_model(model)

if __name__ == "__main__":
    main()
