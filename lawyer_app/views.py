from django.shortcuts import render
import joblib
import pandas as pd

# Load the pre-trained stacking model
stacking_model = joblib.load('lawyer_app/trained.pkl')

# Define a function to load the feature names used during training
def load_feature_names():
    # Load your training data (replace 'shuffled_csv_file.csv' with your actual data file)
    lawyer_df = pd.read_csv('lawyer_app/new_shuffled_csv_file.csv')

    # Perform one-hot encoding for categorical features
    lawyer_df = pd.get_dummies(lawyer_df, columns=['Case Type', 'Specialization', 'Location'])

    # Extract and return the feature column names
    feature_names = lawyer_df.drop(['Lawyer ID', 'Win/Lose'], axis=1).columns.tolist()
    
    return feature_names

# Define a function to load lawyer data including name, specialization, and image URL
def load_lawyer_data():
    # Load your data file that contains Lawyer ID, Lawyer Name, Specialization, and Image URL
    lawyer_data_df = pd.read_csv('lawyer_app/shuffled_csv_file_name.csv')
    return lawyer_data_df

# Define a function to get lawyer details by ID
def get_lawyer_details(lawyer_id):
    # Load lawyer data from 'lawyer_app/shuffled_csv_file_name.csv'
    lawyer_df = load_lawyer_data()

    # Filter lawyer data for the specified lawyer ID
    lawyer_info = lawyer_df[lawyer_df['Lawyer ID'] == lawyer_id]

    if not lawyer_info.empty:
        lawyer_name = lawyer_info.iloc[0]['Lawyer Name']
        specialization = lawyer_info.iloc[0]['Specialization']
        image_url = lawyer_info['Image Url'].iloc[0]  # Corrected here
        return lawyer_name, specialization, image_url

    return None, None, None

# Define the main view for predicting the best lawyer
def predict_best_lawyer(request):
    if request.method == 'POST':
        case_type = request.POST.get('case_type')
        new_location = request.POST.get('Location')
        price = int(request.POST.get('price'))

        # Load the feature names used during training
        feature_names = load_feature_names()

        # Create input data for prediction
        input_data = pd.DataFrame(columns=feature_names)
        input_data.loc[0] = 0
        input_data[f'Case Type_{case_type}'] = 1
        input_data[f'Location_{new_location}'] = 1  # Use 'new_location' here
        input_data['Price'] = price

        # Make a prediction for the specified input data
        prediction = stacking_model.predict(input_data)

        # Assuming you have true labels from your testing data (replace with actual data)
        true_labels = [0]  # Replace with your true labels

        # Identify the best lawyer for the case type
        lawyer_df_case_type = pd.read_csv('lawyer_app/shuffled_csv_file_name.csv')  # Load your data file
        lawyer_df_case_type = pd.get_dummies(lawyer_df_case_type, columns=['Case Type', 'Specialization', 'Location'])
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

        # Fetch details of the best lawyer
        best_lawyer_name, best_lawyer_specialization, best_lawyer_image_url = get_lawyer_details(best_lawyer_id)

        # Display the prediction, the best lawyer, their win/loss score, and accuracy
        if best_lawyer_name is not None:
            best_lawyer_message = f"The best lawyer for the '{case_type}' case is {best_lawyer_name}. Specialization: {best_lawyer_specialization}"
            best_lawyer_image_url = best_lawyer_image_url
        else:
            best_lawyer_message = f"No lawyer found for the '{case_type}' case."
            best_lawyer_image_url = ''

        # Render the result.html template with the result data
        return render(request, 'result.html', {
            'result_message': best_lawyer_message,
            'lawyer_image_url': best_lawyer_image_url,
            'lawyer_name': best_lawyer_name,
            'specialization': best_lawyer_specialization,
        })

    return render(request, 'input.html')  # Adjust the template path as needed
