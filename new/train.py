import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace 'shuffled_csv_file.csv' with your actual data file)
lawyer_df = pd.read_csv('new_shuffled_csv_file.csv')

# Perform one-hot encoding for categorical features
lawyer_df = pd.get_dummies(lawyer_df, columns=['Case Type', 'Specialization', 'Location'])

# Initialize an empty list to store accuracy scores
accuracy_scores = []

# Specify the number of iterations or experiments to run
num_iterations = 10

for _ in range(num_iterations):
    # Split the data into features (X) and target (y)
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

    # Calculate accuracy on the testing data and append it to the list
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracy_scores.append(accuracy)

# Display the accuracy scores as a histogram
plt.figure(figsize=(8, 6))
sns.histplot(accuracy_scores, kde=True)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Accuracy Histogram')
plt.grid(True)

# You can also calculate and display statistics (mean, std, etc.) for the accuracy scores
mean_accuracy = sum(accuracy_scores) / num_iterations
std_accuracy = (sum((x - mean_accuracy) ** 2 for x in accuracy_scores) / (num_iterations - 1)) ** 0.5
print(f"Mean Accuracy: {mean_accuracy:.2f}%")
print(f"Standard Deviation: {std_accuracy:.2f}")

# Show the plots
plt.show()
