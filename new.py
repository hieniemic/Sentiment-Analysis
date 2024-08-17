import streamlit as st
import pickle

# Load the model and vectorizer
with open('best_random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit app
st.title('Comment Sentiment Predictor')
st.write("Type in a comment to predict its sentiment (Positive/Negative)")

# Text input from user
user_input = st.text_area("Enter comment:")

if st.button('Predict'):
    if user_input:
        # Transform the input text using the vectorizer
        X_input = vectorizer.transform([user_input])
        
        # Predict using the loaded model
        prediction = model.predict(X_input)[0]
        
        # Display the result
        st.write(f'The comment is predicted to be: **{prediction}**')
    else:
        st.write("Please enter a comment to predict.")

# Streamlit app
st.title('Comment Sentiment Predictor')
st.write("Type in a comment to predict its sentiment (Positive/Negative)")

# Text input from user
user_input = st.text_area("Enter comment:")

if st.button('Predict'):
    if user_input:
        # Preprocess the user input
        preprocessed_text = convert_unicode(user_input)
        preprocessed_text = process_special_word(preprocessed_text)
        preprocessed_text = normalize_repeated_characters(preprocessed_text)
        # Apply stopwords removal if required
        # preprocessed_text = remove_stopword(preprocessed_text, stopwords_lst)

        # Transform the preprocessed text using the vectorizer
        X_input = vectorizer.transform([preprocessed_text])
        
        # Predict using the loaded model
        prediction = model.predict(X_input)[0]
        
        # Display the result
        st.write(f'The comment is predicted to be: **{prediction}**')
    else:
        st.write("Please enter a comment to predict.")