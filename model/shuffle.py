import pandas as pd

# Load your dataset (replace 'dataset.csv' with your actual dataset file)
df = pd.read_csv('dataset.csv')

# Shuffle the dataset
shuffled_df = df.sample(frac=1.0, random_state=42)

# Save the shuffled dataset to a new CSV file
shuffled_df.to_csv('shuffled_dataset.csv', index=False)

# Confirm that the dataset has been shuffled
print("Dataset shuffled and saved as 'shuffled_dataset.csv'")
