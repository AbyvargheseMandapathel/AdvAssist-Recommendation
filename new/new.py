import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
import joblib

# Load the dataset
lawyer_df = pd.read_csv('new_shuffled_csv_file.csv')

# Perform one-hot encoding for categorical features
lawyer_df = pd.get_dummies(lawyer_df, columns=['Case Type', 'Specialization', 'Location'])

# Split the dataset into features (X) and target (y)
X = lawyer_df.drop(['Lawyer ID', 'Win/Lose'], axis=1)
y = lawyer_df['Win/Lose']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize base classifiers
decision_tree = DecisionTreeClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)

# Create a list of base estimators
estimators = [('decision_tree', decision_tree), ('knn', knn)]

# Initialize the stacking classifier with a final estimator
stacking_model = StackingClassifier(estimators=estimators, final_estimator=DecisionTreeClassifier())

# Train the stacking model on the training data
stacking_model.fit(X_train, y_train)

# Specify the case type, price, and location for which you want to make a prediction
case_type = "Murder"
price = 231  # Replace with the desired price
location = "Vilage Court"  # Replace with the desired location

# Create input data for prediction
input_data = pd.DataFrame(columns=X.columns)
input_data.loc[0] = 0
input_data[f'Case Type_{case_type}'] = 1
input_data['Price'] = price
input_data[f'Location_{location}'] = 1  # One-hot encode the location

# Make a prediction for the specified input data
prediction = stacking_model.predict(input_data)

# Get the win probability for each lawyer
lawyer_df['Win Probability'] = stacking_model.predict_proba(X)[:, 1]

# Find the lawyer with the highest win probability
best_lawyer = lawyer_df.loc[lawyer_df['Win Probability'].idxmax()]

# Display the prediction
if prediction[0] == 1:
    print(f"The predicted outcome for the case is 'Win'.")
else:
    print(f"The predicted outcome for the case is 'Lose'.")

# Display the best lawyer and their win probability
print(f"The best lawyer for the case is Lawyer {best_lawyer['Lawyer ID']} with a win probability of {best_lawyer['Win Probability']:.2%}")
