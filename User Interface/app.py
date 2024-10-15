from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS
import numpy as np 

app = Flask(__name__)

# Load the trained model
model = joblib.load(r'C:\Users\Jigar Prajapati\Downloads\DSML_project\DSML_project\models\model2.pkl')

# Enable CORS for your Flask app
CORS(app)

# Encoding mappings
encode = {
    'Total_employees': [['26-100', '6-25', '0', 'More than 1000', '100-500', '500-1000', '1-5'], [3, 2, 0, 6, 4, 5, 1]],
    'Mental_health_coverage': [['Not eligible for coverage / N/A', 'No', 'Yes', "I don't know"], [0, 2, 3, 1]],
    'Awareness_of_options_under_coverage': [['Yes', 'No', 'I am not sure'], [2, 1, 0]],
    'Employers_discussion_on_mental_health': [['Yes', 'No', "I don't know"], [2, 1, 0]],
    'Resources_and_options_for_help': [['Yes', 'No', "I don't know"], [2, 1, 0]],
    'Anomity_of_employees_using_mental_health_resources': [['Yes', 'No', "I don't know"], [2, 1, 0]],
    'Mental_issue_related_leave': [['Somewhat easy', 'Neither easy nor difficult', 'Very easy', 'Very difficult', 'Somewhat difficult', "I don't know"], [4, 2, 5, 0, 1, 3]],
    'Discussing_mental_health_neg_consequences': [['No', 'Yes', 'Maybe'], [2, 0, 1]],
    'Discussing_physical_health_neg_consequences': [['No', 'Maybe', 'Yes'], [2, 1, 0]],
    'Comfort_discussing_with_coworkers': [['Maybe', 'Yes', 'No'], [1, 2, 0]],
    'Comfort_discussing_with_supervisor': [['Yes', 'No', 'Maybe'], [2, 0, 1]],
    'Employer_seriousness_on_mental_health': [['Yes', 'No', "I don't know"], [2, 0, 1]],
    'Negative_consequences_of_coworkers_with_mental_health_disorder': [['No', 'Yes'], [1, 0]],
    'prev_empl_mental_benefits': [['Yes, they all did', "I don't know", 'Some did', 'No, none did', 'Not Applicable'], [4, 2, 3, 0, 1]],
    'aware_benefits_prev_empl': [['I was aware of some', 'N/A (not currently aware)', 'Yes, I was aware of all of them', 'No, I only became aware later', 'Not Applicable'], [3, 2, 4, 0, 1]],
    'formal_discuss_mental_health_premp': [['None did', 'Some did', 'Not Applicable', "I don't know", 'Yes, they all did'], [0, 3, 1, 2, 4]],
    'premp_provide_resources_learn_seekhelp': [['Some did', 'None did', 'Not Applicable', 'Yes, they all did'], [2, 0, 1, 3]],
    'anonymity_protect_premp_choose_resources': [['Yes, always', "I don't know", 'Sometimes', 'No', 'Not Applicable'], [4, 2, 3, 0, 1]],
    'neg_eff_discuss_mental_health_premp': [['None of them', 'Some of them', 'Yes, all of them', "I don't know", 'Not Applicable'], [4, 1, 0, 2, 3]],
    'neg_eff_discuss_phy_health_premp': [['None of them', 'Some of them', 'Yes, all of them', 'Not Applicable'], [3, 1, 0, 2]],
    'willing_discuss_issue_prev_coworkers': [['No, at none of my previous employers', 'Some of my previous employers', 'Yes, at all of my previous employers', 'Not Applicable'], [0, 1, 3, 2]],
    'willing_discuss_issue_dir_supervisors': [['Some of my previous employers', 'Yes, at all of my previous employers', 'No, at none of my previous employers', "I don't know", 'Not Applicable'], [3, 4, 0, 1, 2]],
    'premp_took_menhealth_seriously_as_phyhealth': [["I don't know", 'Some did', 'None did', 'Yes, they all did', 'Not Applicable'], [0, 2, 1, 3, 0]],
    'neg_comments_abt_coworkers_with_menissues_prev_workplace': [['None of them', 'Some of them', 'Not Applicable', 'Yes, all of them'], [1, 2, 0, 3]],
    'bring_phyhealth_issue_with_potential_empl_interview': [['Maybe', 'Yes', 'No'], [1, 2, 0]],
    'bring_menhealth_issue_with_potential_empl_interview': [['Maybe', 'No', 'Yes'], [1, 0, 2]],
    'hurt_carrer_if_identified_as_person_with_mental_issue': [['Maybe', "No, I don't think it would", 'Yes, I think it would', 'No, it has not', 'Yes, it has'], [1, 0, 2, 0, 2]],
    'more_neg_view_from_coworker_if_know_abt_men_issue': [["No, I don't think they would", 'Maybe', 'Yes, they do', 'Yes, I think they would', 'No, they do not'], [0, 1, 2, 2, 0]],
    'share_w_friends_fam': [['Somewhat open', 'Neutral', 'Not applicable to me (I do not have a mental illness)', 'Very open', 'Not open at all', 'Somewhat not open'], [4, 3, 0, 5, 1, 2]],
    'workplace_response': [['No', 'Maybe/Not sure', 'Yes, I experienced', 'Yes, I observed', np.nan], [0, 1, 2, 2, 1]],
    'fam_hist': [['No', 'Yes', "I don't know"], [0, 2, 1]],
    'past_mh': [['Yes', 'Maybe', 'No'], [2, 1, 0]],
    'curr_mh': [['No', 'Yes', 'Maybe'], [0, 2, 1]],
    'diagnosed_prof': [['Yes', 'No'], [1, 0]],
    'work_interference_treatment': [['Not applicable to me', 'Rarely', 'Sometimes', 'Never', 'Often'], [0, 2, 3, 1, 4]],
    'work_interference_no_treatment': [['Not applicable to me', 'Sometimes', 'Often', 'Rarely', 'Never'], [0, 3, 4, 2, 1]],
    'remote': [['Sometimes', 'Never', 'Always'], [1, 0, 2]]
}

# # Reverse decoding mappings for prediction result interpretation
decode = {
    0: 'No',
    1: 'Maybe',
    2: 'Yes',
    
    # Self_employed
    0: 'Not Self-Employed',

    # Total_employees
    1: '1-5',
    2: '6-25',
    3: '26-100',
    4: '100-500',
    5: '500-1000',
    6: 'More than 1000',

    # Tech_organization
    0: 'Non-Tech Organization',
    1: 'Tech Organization',

    # Mental_health_coverage
    0: 'Not eligible for coverage / N/A',
    1: "I don't know",
    2: 'No',
    3: 'Yes',

    # Awareness_of_options_under_coverage
    0: 'No',
    1: 'Yes',
    2: 'I am not sure',

    # Employers_discussion_on_mental_health
    0: "I don't know",
    1: 'No',
    2: 'Yes',

    # Resources_and_options_for_help
    0: "I don't know",
    1: 'No',
    2: 'Yes',

    # Anomity_of_employees_using_mental_health_resources
    0: "I don't know",
    1: 'No',
    2: 'Yes',

    # Mental_issue_related_leave
    0: 'Very difficult',
    1: 'Somewhat difficult',
    2: 'Neither easy nor difficult',
    3: "I don't know",
    4: 'Somewhat easy',
    5: 'Very easy',

    # work_interference_treatment
    0: 'Not applicable to me',
    1: 'Never',
    2: 'Rarely',
    3: 'Sometimes',
    4: 'Often',

    # work_interference_no_treatment
    0: 'Not applicable to me',
    1: 'Never',
    2: 'Rarely',
    3: 'Sometimes',
    4: 'Often',

    # remote
    0: 'Never',
    1: 'Sometimes',
    2: 'Always'
}


# States mapping for USA regions
north_east = ['Pennsylvania', 'New York', 'Rhode Island', 'Maine', 'New Jersey', 'New Hampshire', 'Massachusetts', 'Vermont', 'Connecticut']
south = ['Delaware', 'District of Columbia', 'Texas', 'Louisiana', 'Oklahoma', 'Alabama', 'Kentucky', 'Virginia', 'South Carolina', 'Maryland', 'West Virginia', 'North Carolina', 'Georgia', 'Florida', 'Tennessee']
mid_west = ['Illinois', 'Indiana', 'Minnesota', 'Iowa', 'Ohio', 'Michigan', 'Wisconsin', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota', 'Kansas']
west = ['California', 'Montana', 'Oregon', 'New Mexico', 'Alaska', 'Washington', 'Nevada', 'Arizona', 'Utah', 'Colorado', 'Hawaii', 'Idaho']

# Function to apply encoding to user input
def apply_encoding(input_data):
    # Manual encoding using predefined mappings
    for col, mapping in encode.items():
        if col in input_data:
            if input_data[col] in mapping[0]:
                input_data[col] = mapping[1][mapping[0].index(input_data[col])]
            else:
                input_data[col] = -1  # Handle unseen categories with a default value

    # Encode us_state_live and us_state_work
    if 'us_state_live' in input_data:
        if input_data['us_state_live'] in north_east:
            input_data['us_state_live'] = 1
        elif input_data['us_state_live'] in mid_west:
            input_data['us_state_live'] = 2
        elif input_data['us_state_live'] in west:
            input_data['us_state_live'] = 3
        elif input_data['us_state_live'] in south:
            input_data['us_state_live'] = 4
        else:
            input_data['us_state_live'] = 0  # Default for unknown states

    if 'us_state_work' in input_data:
        if input_data['us_state_work'] in north_east:
            input_data['us_state_work'] = 1
        elif input_data['us_state_work'] in mid_west:
            input_data['us_state_work'] = 2
        elif input_data['us_state_work'] in west:
            input_data['us_state_work'] = 3
        elif input_data['us_state_work'] in south:
            input_data['us_state_work'] = 4
        else:
            input_data['us_state_work'] = 0  # Default for unknown states

    # Encode country_live and country_work
    if 'country_live' in input_data:
        input_data['country_live'] = 0 if input_data['country_live'] == 'USA' else 1
    if 'country_work' in input_data:
        input_data['country_work'] = 0 if input_data['country_work'] == 'USA' else 1

    # Handle remaining string columns by converting them to category codes
    for key in input_data:
        if isinstance(input_data[key], str):
            input_data[key] = pd.Series([input_data[key]]).astype('category').cat.codes[0]

    return input_data

# Function to decode the prediction result back to a readable format
def decode_prediction(prediction):
    return decode.get(prediction, "Unknown")

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/mentalHealth')
def test():
    return render_template('mentalHealth.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the form
    user_input = request.form.to_dict()

    # Apply encoding to the user input
    encoded_input = apply_encoding(user_input)

    # Convert the encoded input into a DataFrame
    input_df = pd.DataFrame([encoded_input])

    # Make the prediction using the model
    prediction = model.predict(input_df)

    # Decode the prediction into a readable format
    # decoded_result = decode_prediction(prediction[0])

    # # Display the result
    # return render_template('index.html', prediction_text=f'The predicted result is: {decoded_result}')

if __name__ == '__main__':
    app.run(debug=True)
