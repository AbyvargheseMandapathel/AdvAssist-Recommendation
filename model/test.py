import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace 'shuffled_dataset.csv' with the actual filename)
data = pd.read_csv('shuffled_dataset.csv')

# Replace 'Case Type', 'Lawyer Name', and 'Win/Lose' with your actual column names if different
X = data[['Case Type', 'Lawyer Name']]
y = data['Win/Lose']

# Encode categorical variables
le_case_type = LabelEncoder()
le_lawyer_name = LabelEncoder()

# Fit and transform 'Case Type' and 'Lawyer Name' columns
X['Case Type'] = le_case_type.fit_transform(X['Case Type'])
X['Lawyer Name'] = le_lawyer_name.fit_transform(X['Lawyer Name'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
knn = KNeighborsClassifier()

# Create a stacking ensemble of Decision Tree, Random Forest, and KNN
estimators = [('decision_tree', decision_tree), ('random_forest', random_forest), ('knn', knn)]
stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=decision_tree)

# Define the parameter grid for hyperparameter tuning (for Decision Tree)
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object for Decision Tree
grid_search_decision_tree = GridSearchCV(
    decision_tree, param_grid_decision_tree, cv=5, n_jobs=-1, verbose=2
)

# Fit the models
grid_search_decision_tree.fit(X_train, y_train)
stacking_classifier.fit(X_train, y_train)

# Predictions
y_decision_tree_pred = grid_search_decision_tree.predict(X_test)
y_stacking_classifier_pred = stacking_classifier.predict(X_test)

# Evaluate Decision Tree
decision_tree_accuracy = accuracy_score(y_test, y_decision_tree_pred)
print(f'Decision Tree Accuracy: {decision_tree_accuracy:.2%}')

# Evaluate Stacking Classifier
stacking_classifier_accuracy = accuracy_score(y_test, y_stacking_classifier_pred)
print(f'Stacking Classifier Accuracy: {stacking_classifier_accuracy:.2%}')

# Decode lawyer names from numeric values
X_train['Lawyer Name'] = le_lawyer_name.inverse_transform(X_train['Lawyer Name'])

# Find the best lawyer for a given case type
def find_best_lawyer(case_type):
    case_type_encoded = le_case_type.transform([case_type])[0]
    # Use a placeholder value (-1) for Lawyer Name when making predictions
    predicted_outcome = stacking_classifier.predict([[case_type_encoded, -1]])
    best_lawyers = X_train[(X_train['Case Type'] == case_type_encoded) & (y_train == predicted_outcome[0])]
    return best_lawyers['Lawyer Name'].tolist()

# Example: Find the best lawyer for a case type 'Criminal'
best_lawyers_for_Deportation = find_best_lawyer('Deportation')
print(f'Best lawyers for Deportation cases: {best_lawyers_for_Deportation}')
