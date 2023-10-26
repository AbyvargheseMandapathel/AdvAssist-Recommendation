import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset (replace 'shuffled_csv_file.csv' with your actual data file)
lawyer_df = pd.read_csv('new_shuffled_csv_file.csv')

# Check for missing values and impute them with the mean
lawyer_df = lawyer_df.fillna(lawyer_df.mean())


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

# Make predictions on the testing data
y_pred = stacking_model.predict(X_test)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Lose", "Win"], yticklabels=["Lose", "Win"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Calculate accuracy on the testing data
accuracy = accuracy_score(y_test, y_pred) * 100

# Display the accuracy as a graph
plt.figure(figsize=(8, 6))
plt.plot([accuracy] * len(y_test), label=f'Accuracy: {accuracy:.2f}%')
plt.xlabel('Test Samples')
plt.ylabel('Accuracy')
plt.title('Accuracy on Testing Data')
plt.legend()
plt.grid(True)

# Display the accuracy as a histogram
plt.figure(figsize=(8, 6))
sns.histplot([accuracy], kde=True)  # Wrap accuracy in a list or array
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Accuracy Histogram')
plt.grid(True)

# Specify the case type, price, and location for which you want to make a prediction
case_type = input("Enter the case type: ")
price = int(input("Enter the price: "))
location = input("Enter the location: ")

# Create input data for prediction
input_data = pd.DataFrame(columns=X.columns)
input_data.loc[0] = 0
input_data[f'Case Type_{case_type}'] = 1
input_data['Price'] = price
input_data[f'Location_{location}'] = 1  # One-hot encode the location

# Make a prediction for the specified input data
prediction = stacking_model.predict(input_data)

# Identify the best lawyer for the case type
lawyer_df_case_type = lawyer_df[lawyer_df[f'Case Type_{case_type}'] == 1]
lawyer_ids = lawyer_df_case_type['Lawyer ID'].unique()
best_lawyer_id = None
best_score = -1

for lawyer_id in lawyer_ids:
    lawyer_cases = lawyer_df_case_type[lawyer_df_case_type['Lawyer ID'] == lawyer_id]
    total_cases = len(lawyer_cases)


    if total_cases == 0:
        continue

    wins = lawyer_cases['Win/Lose'].sum()
    losses = total_cases - wins
    score = wins if prediction[0] == 1 else losses

    if score > best_score:
        best_score = score
        best_lawyer_id = lawyer_id

# Display the prediction, the best lawyer, their win/loss score, and accuracy
if best_lawyer_id is not None:
    print(f"The best lawyer for the '{case_type}' case is Lawyer {best_lawyer_id}")
else:
    print(f"No lawyer found for the '{case_type}' case.")

# After making predictions (y_pred) and having true labels (y_test), calculate the metrics
accuracy = accuracy_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100

# Print the results
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")

# Save the stacking model as a .pkl file
model_filename = "trainednew.pkl"
joblib.dump(stacking_model, model_filename)

print(f"Stacking model saved to {model_filename}")

# Show the plots
plt.show()
