#This section learned from: https://www.datacamp.com/tutorial/decision-tree-classification-python

import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


data = pd.read_csv("../data/training_data_simple.csv")

X = data[
    [
        "points_this_week",
        "attendance_rate",
        "behavior_referrals"
    ]
]

y = data["needs_reinforcement"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)


model = DecisionTreeClassifier(
    random_state=1,
    max_depth=5,
    min_samples_leaf=5,
    class_weight="balanced"  #stops tree from overfitting
)


model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
classify = classification_report(y_test, predictions, zero_division=0)

print(classify)
print(predictions)
print(accuracy)
print(precision)
print(recall)
print(f1)

joblib.dump(model, "student_model.pkl")