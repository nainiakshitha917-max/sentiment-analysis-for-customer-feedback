import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle


# -----------------------------
# Step 1: Create sample dataset
# -----------------------------

data = {
    "feedback": [
        "The product is excellent",
        "I am very happy with the service",
        "Amazing product and fast delivery",
        "The quality is very good",
        "I love this product",
        "Very satisfied with my purchase",
        "The service was excellent",
        "Good product and good quality",

        "The product is terrible",
        "I am very unhappy with the service",
        "Very bad experience",
        "The quality is poor",
        "I hate this product",
        "The delivery was very late",
        "The service was horrible",
        "I am disappointed with the product",

        "The product is okay",
        "The service was average",
        "It is neither good nor bad",
        "The product is normal",
        "Average quality",
        "The experience was okay",
        "Delivery was normal",
        "I have no strong opinion"
    ],

    "sentiment": [
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",
        "Positive",

        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",
        "Negative",

        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral"
    ]
}

df = pd.DataFrame(data)


# -----------------------------
# Step 2: Split the dataset
# -----------------------------

X = df["feedback"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Step 3: Create ML pipeline
# -----------------------------

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


# -----------------------------
# Step 4: Train the model
# -----------------------------

model.fit(X_train, y_train)


# -----------------------------
# Step 5: Test the model
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# -----------------------------
# Step 6: Save the model
# -----------------------------

with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")