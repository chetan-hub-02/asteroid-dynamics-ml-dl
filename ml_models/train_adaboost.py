import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the datasets
class_1 = pd.read_csv('dataset/ML_data/train_test_split_class_1.csv')
class_2 = pd.read_csv('dataset/ML_data/train_test_split_class_2.csv')

# Take a sample of 15,300 elements from class_2
class_2_sample = class_2.sample(n=15300, random_state=42)

# Combine the two datasets
data = pd.concat([class_1, class_2_sample])

# Define features and target variable
features = ['Semi Major Axis', 'Eccentricity', 'Inclination', 'Perihelion Arg', 
            'Asc Node Longitude', 'Mean Anomaly', 'Perihelion Distance', 'Asteroid Type']
target = 'Class'

X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# Base estimator (weak learner)
dt = DecisionTreeClassifier(max_depth=1, random_state=42)

# AdaBoost model
ab = AdaBoostClassifier(base_estimator=dt, random_state=42)

# Hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'base_estimator__max_depth': [1, 3, 5],
    'algorithm': ['SAMME', 'SAMME.R']
}

# Reproducible Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search
grid_search_ab = GridSearchCV(
    estimator=ab,
    param_grid=param_grid,
    cv=cv,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search_ab.fit(X_train, y_train)

# Best model
best_ab_model = grid_search_ab.best_estimator_

# Save best model
joblib.dump(best_ab_model, "best_adaboost_model.pkl")

# Best hyperparameters
print("\nBest Hyperparameters:")
print(grid_search_ab.best_params_)

# (Optional but recommended)
print(f"\nBest CV Accuracy: {grid_search_ab.best_score_:.4f}")

# Training Evaluation
y_train_pred = best_ab_model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)
train_confusion = confusion_matrix(y_train, y_train_pred)

print("\nTraining Metrics:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"F1 Score: {train_f1}")
print(f"Confusion Matrix:\n{train_confusion}")

# Test Evaluation
y_test_pred = best_ab_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1}")
print(f"Confusion Matrix:\n{test_confusion}")
