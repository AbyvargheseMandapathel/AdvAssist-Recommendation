from django.shortcuts import render
import joblib
import pandas as pd

# Load the pre-trained stacking model
stacking_model = joblib.load('lawyer_app\stacking_model.pkl')

# Define a function to load the feature names used during training
def load_feature_names():
    # Load your training data (replace 'shuffled_csv_file.csv' with your actual data file)
    lawyer_df = pd.read_csv('lawyer_app\shuffled_csv_file.csv')

    # Perform one-hot encoding for categorical features
    lawyer_df = pd.get_dummies(lawyer_df, columns=['Case Type', 'Specialization'])

    # Extract and return the feature column names
    feature_names = lawyer_df.drop(['Lawyer ID', 'Win/Lose'], axis=1).columns.tolist()
    
    return feature_names

# Define a function to load lawyer names
def load_lawyer_names():
    # Load your data file that contains Lawyer ID and Lawyer Name mapping
    lawyer_name_df = pd.read_csv('lawyer_app\shuffled_csv_file_name.csv')
    lawyer_name_mapping = dict(zip(lawyer_name_df['Lawyer ID'], lawyer_name_df['Lawyer Name']))
    return lawyer_name_mapping

def predict_best_lawyer(request):
    if request.method == 'POST':
        case_type = request.POST.get('case_type')

        # Load the feature names used during training
        feature_names = load_feature_names()

        # Create input data for prediction
        input_data = pd.DataFrame(columns=feature_names)
        input_data.loc[0] = 0
        input_data[f'Case Type_{case_type}'] = 1

        # Make a prediction for the specified case type
        prediction = stacking_model.predict(input_data)

        # Identify the best lawyer for the case type
        lawyer_df_case_type = pd.read_csv('lawyer_app\shuffled_csv_file.csv')  # Load your data file
        lawyer_df_case_type = pd.get_dummies(lawyer_df_case_type, columns=['Case Type', 'Specialization'])
        lawyer_df_case_type = lawyer_df_case_type[
            lawyer_df_case_type[f'Case Type_{case_type}'] == 1
        ]

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

        # Load lawyer names
        lawyer_name_mapping = load_lawyer_names()

        # Get the lawyer's name based on their ID
        best_lawyer_name = lawyer_name_mapping.get(best_lawyer_id, "Unknown Lawyer")

        # Prepare the result message
        if best_lawyer_id is not None:
            best_lawyer_message = f"The best lawyer for the '{case_type}' case is {best_lawyer_name}"
        else:
            best_lawyer_message = f"No lawyer found for the '{case_type}' case."

        # Render the result.html template with the result data
        return render(request, 'result.html', {
            'result_message': best_lawyer_message,
        })

    return render(request, 'input.html')  # Adjust the template path as needed
