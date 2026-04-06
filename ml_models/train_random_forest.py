import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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

# Perform 80:20 split for training and testing with shuffling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
)

# Set up Random Forest model
rf = RandomForestClassifier(random_state=42)

# Refined hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [2, 4, 8],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2']
}

# Reproducible Stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search with CV
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search_rf.fit(X_train, y_train)

# Get the best model from the grid search
best_rf_model = grid_search_rf.best_estimator_

# Save best model
joblib.dump(best_rf_model, "best_random_forest_model.pkl")

# Print the best hyperparameters
print("\nBest Hyperparameters:")
print(grid_search_rf.best_params_)

# (Optional but useful)
print(f"\nBest CV Accuracy: {grid_search_rf.best_score_:.4f}")

# Training Evaluation
y_train_pred = best_rf_model.predict(X_train)

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
y_test_pred = best_rf_model.predict(X_test)

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
