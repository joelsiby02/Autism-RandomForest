from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Define the questions for each age group
questions = {
    "children": [
        "Does your child avoid eye contact?",
        "Does your child show interest in other children?",
        "Does your child engage in pretend play?",
        "Does your child repeat the same actions or words?",
        "Does your child have trouble understanding others' feelings?",
        "Does your child have trouble adjusting to changes in routine?",
        "Does your child flap their hands or spin objects?",
        "Does your child have difficulty with social interactions?",
        "Does your child appear oversensitive to noises or lights?",
        "Does your child have specific interests or routines?"
    ],
    "adolescents": [
        "Do you have difficulty maintaining friendships?",
        "Do you prefer to spend time alone rather than with friends?",
        "Do you find it hard to understand what others are thinking or feeling?",
        "Do you have trouble understanding jokes or sarcasm?",
        "Do you find it difficult to adapt to changes?",
        "Do you engage in repetitive behaviors or have specific routines?",
        "Do you have intense interests in specific topics?",
        "Do you find social situations overwhelming?",
        "Do you avoid eye contact?",
        "Do you have difficulty with small talk or casual conversations?"
    ],
    "young_adults": [
        "Do you find it difficult to understand other people's emotions?",
        "Do you prefer to follow routines and find change difficult?",
        "Do you avoid social situations or find them overwhelming?",
        "Do you struggle with making eye contact during conversations?",
        "Do you have specific hobbies or interests that you focus on intensely?",
        "Do you find it challenging to engage in small talk?",
        "Do you often miss social cues, such as when someone is being sarcastic?",
        "Do you feel uncomfortable in group settings?",
        "Do you prefer to be alone rather than with others?",
        "Do you find loud noises or bright lights distressing?"
    ],
    "adults": [
        "Do you often prefer to be alone rather than with others?",
        "Do you find it challenging to understand social cues?",
        "Do you have specific routines or rituals that you prefer not to break?",
        "Do you avoid eye contact in conversations?",
        "Do you feel overwhelmed in social situations?",
        "Do you have intense interests or hobbies?",
        "Do you struggle with changes to your routine?",
        "Do you find it hard to make friends?",
        "Do you have difficulty understanding jokes or sarcasm?",
        "Do you feel uncomfortable in new situations or with unfamiliar people?"
    ]
}

# Define the function at the global scope
def get_model_and_scaler_for_age(age):
    if 0 < age <= 10:
        model_path = 'models/children_asd_model.pkl'
        age_group = 'children'
    elif 11 <= age <= 17:
        model_path = 'models/adolescent_asd_model.pkl'
        age_group = 'adolescents'
    elif 18 <= age <= 35:
        model_path = 'models/young_asd_model.pkl'
        age_group = 'young_adults'
    else:
        model_path = 'models/adult_asd_model.pkl'
        age_group = 'adults'
    
    model = joblib.load(model_path)
    scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
    
    return model, scaler, age_group

@app.route('/get_questions', methods=['GET'])
def get_questions():
    age = int(request.args.get('age'))
    
    if 0 < age <= 10:
        age_group = 'children'
    elif 11 <= age <= 17:
        age_group = 'adolescents'
    elif 18 <= age <= 35:
        age_group = 'young_adults'
    else:
        age_group = 'adults'
    
    return jsonify({
        'age_group': age_group,
        'questions': questions[age_group]
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    try:
        age = int(data['age'])
        responses = data['responses']
        categorical_values = data.get('categorical_values', {})  # Ensure categorical inputs are received
    except KeyError:
        return jsonify({'error': 'Missing parameters'}), 400
    
    # Convert responses to integers
    try:
        user_responses = [int(r) for r in responses]
    except ValueError:
        return jsonify({'error': 'Invalid response format'}), 400
    
    # Load model and scaler
    model, scaler, age_group = get_model_and_scaler_for_age(age)

    # Define feature names (matching your dataset)
    feature_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient']
    
    # Add categorical feature names
    categorical_feature_names = ['Sex', 'Jaundice', 'Family_mem_with_ASD']
    feature_names.extend(categorical_feature_names)

    # Convert categorical values (Example: Label Encoding)
    encoded_categorical_values = []
    for feature in categorical_feature_names:
        value = categorical_values.get(feature, 'unknown')  # Default to 'unknown' if missing
        if feature == 'Sex':  # Encode Male as 1, Female as 0
            encoded_categorical_values.append(1 if value.lower() == 'm' else 0)
        elif feature in ['Jaundice', 'Family_mem_with_ASD']:
            encoded_categorical_values.append(1 if value.lower() == 'yes' else 0)
        else:
            encoded_categorical_values.append(0)  # Default encoding
    
    # Combine numeric and categorical features
    user_input = user_responses + encoded_categorical_values

    # Convert to DataFrame
    user_responses_df = pd.DataFrame([user_input], columns=feature_names)

    # Scale and predict
    user_responses_scaled = scaler.transform(user_responses_df)
    prediction = model.predict(user_responses_scaled)
    
    return jsonify({
        'prediction': int(prediction[0]),
        'age_group': age_group
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
