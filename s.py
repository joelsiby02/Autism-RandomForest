import streamlit as st
import requests

# API configuration
API_BASE_URL = "http://localhost:5000"

def main():
    st.title("ASD Traits Assessment")
    
    # Age input
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=18, step=1)

    # Relevant categorical inputs (only those in your dataset)
    gender = st.radio("Select your gender:", ["Male", "Female"])
    jaundice = st.radio("Did you have jaundice at birth?", ["Yes", "No"])
    family_asd_history = st.radio("Is there a family history of ASD?", ["Yes", "No"])

    # Convert categorical inputs to numerical values
    categorical_data = {
        "Sex": 1 if gender == "Male" else 0,
        "Jaundice": 1 if jaundice == "Yes" else 0,
        "Family_mem_with_ASD": 1 if family_asd_history == "Yes" else 0,
    }

    # Get questions from backend
    questions_response = requests.get(f"{API_BASE_URL}/get_questions", params={"age": age})
    
    if questions_response.status_code != 200:
        st.error("Failed to load questions")
        return
    
    questions_data = questions_response.json()
    age_group = questions_data['age_group'].replace('_', ' ')
    questions = questions_data['questions']
    
    # Display questions
    st.subheader(f"Questions for {age_group}")
    
    responses = []
    with st.form("assessment_form"):
        for i, question in enumerate(questions, 1):
            response = st.radio(
                question,
                options=("Yes", "No"),
                index=None,
                key=f"q{i}"
            )
            responses.append(1 if response == "Yes" else 0 if response == "No" else None)
        
        submitted = st.form_submit_button("Submit Assessment")
    
    if submitted:
        # Validate responses
        if None in responses:
            st.error("Please answer all questions before submitting")
            return
        
        # Send prediction request with categorical variables included
        prediction_data = {
            "age": age,
            "responses": responses,
            "categorical_features": categorical_data
        }
        
        prediction_response = requests.post(
            f"{API_BASE_URL}/predict",
            json=prediction_data
        )
        
        if prediction_response.status_code == 200:
            result = prediction_response.json()
            if result['prediction'] == 1:
                st.error("Your responses suggest possible ASD traits. Please consult a healthcare professional.")
            else:
                st.success("Based on your responses, ASD traits are not typically present. Consult a professional for medical advice.")
        else:
            error = prediction_response.json().get('error', 'Unknown error')
            st.error(f"Prediction failed: {error}")

if __name__ == "__main__":
    main()
