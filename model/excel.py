import pandas as pd
import random

# Define lawyer specializations and corresponding case types
specializations = {
    'Criminal Lawyer': ['Murder', 'Assault', 'Robbery', 'Drug', 'Burglary'],
    'Family Lawyer': ['Divorce', 'ChildCustody', 'Adoption'],
    'Tax Lawyer': ['TaxEvasion', 'TaxFraud', 'IRSAudit'],
    'Business Lawyer': ['ContractDispute', 'BusinessFraud', 'IntellectualProperty'],
    'Immigration Lawyer': ['VisaApplication', 'Deportation', 'Asylum']
}

# Initialize empty lists to store data
data = {'Specialization': [], 'Case Type': [], 'Case Number': [], 'Lawyer ID': [], 'Lawyer Name': [], 'Win/Lose': []}

# Define the lawyer IDs and names for each specialization
lawyers_data = {
    'Criminal Lawyer': [(1000, 'John Smith'), (1001,Lawyer1, 'Alice Johnson')],
    'Family Lawyer': [(1001,Lawyer2, 'David Brown'), (1001,Lawyer3, 'Emma White')],
    'Immigration Lawyer': [(1001,Lawyer4, 'Michael Wilson'), (1001,Lawyer5, 'Sophia Martinez')],
    'Business Lawyer': [(1001,Lawyer6, 'Daniel Lee')],
    'Tax Lawyer': [(1001,Lawyer7, 'Olivia Davis')]
}

# Define the minimum and maximum number of cases per lawyer
min_cases_per_lawyer = 40
max_cases_per_lawyer = 60

# Generate data for each lawyer
for lawyer_specialization, lawyers_info in lawyers_data.items():
    for lawyer_id, lawyer_name in lawyers_info:
        lawyer_cases = specializations[lawyer_specialization]  # Get matching case types

        num_cases = random.randint(min_cases_per_lawyer, max_cases_per_lawyer)

        for case_id in range(1, num_cases + 1):
            case_type = random.choice(lawyer_cases)
            win_lose = random.choice([0, 1])

            lawyer_data = {
                'Specialization': lawyer_specialization,
                'Case Type': case_type,
                'Case Number': f'{1000 + case_id}',
                'Lawyer ID': f'{lawyer_id}',
                'Lawyer Name': lawyer_name,
                'Win/Lose': win_lose
            }

            data['Specialization'].append(lawyer_data['Specialization'])
            data['Case Type'].append(lawyer_data['Case Type'])
            data['Case Number'].append(lawyer_data['Case Number'])
            data['Lawyer ID'].append(lawyer_data['Lawyer ID'])
            data['Lawyer Name'].append(lawyer_data['Lawyer Name'])
            data['Win/Lose'].append(lawyer_data['Win/Lose'])

# Create a DataFrame from the generated data with the desired column order
lawyer_df = pd.DataFrame(data, columns=['Specialization', 'Case Type', 'Case Number', 'Lawyer ID', 'Lawyer Name', 'Win/Lose'])

# Save the dataset to a CSV file
lawyer_df.to_csv('dataset.csv', index=False)

# Calculate the winning probability for each lawyer
def calculate_winning_probability(lawyer_id):
    lawyer_cases = lawyer_df[lawyer_df['Lawyer ID'] == lawyer_id]
    total_cases = len(lawyer_cases)
    if total_cases == 0:
        return 0  # Avoid division by zero
    wins = lawyer_cases['Win/Lose'].sum()
    winning_probability = int((wins / total_cases) * 100)  # Convert to percentage
    return winning_probability

# Calculate the winning probability for each lawyer
unique_lawyer_ids = lawyer_df['Lawyer ID'].unique()
winning_probabilities = []
for lawyer_id in unique_lawyer_ids:
    winning_probability = calculate_winning_probability(lawyer_id)
    winning_probabilities.append({'Lawyer ID': lawyer_id, 'Winning Probability': winning_probability})

# Create a DataFrame to store the winning probabilities
winning_probabilities_df = pd.DataFrame(winning_probabilities)

# Find the lawyer with the highest winning probability
best_lawyer = winning_probabilities_df.loc[winning_probabilities_df['Winning Probability'].idxmax()]

# Print the best lawyer, their name, and their winning probability
print(f"Best Lawyer: {lawyer_df.loc[lawyer_df['Lawyer ID'] == best_lawyer['Lawyer ID']]['Lawyer Name'].values[0]} (Lawyer {best_lawyer['Lawyer ID']}) with a winning probability of {best_lawyer['Winning Probability']}%")
