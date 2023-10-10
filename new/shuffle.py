import pandas as pd
import random

# Load the CSV file (replace 'your_csv_file.csv' with your actual file name)
csv_file = 'dataset.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Get the header row
header = df.columns.tolist()

# Exclude the header row for shuffling
data_rows = df.values.tolist()
data_rows = random.sample(data_rows, len(data_rows))  # Shuffle the data rows

# Create a new DataFrame with the shuffled data and original headers
shuffled_df = pd.DataFrame(data_rows, columns=header)

# Save the shuffled DataFrame to a new CSV file (replace 'shuffled_csv_file.csv' with your desired output file name)
output_file = 'new_shuffled_csv_file.csv'
shuffled_df.to_csv(output_file, index=False)

print(f"Shuffled data saved to {output_file}")
