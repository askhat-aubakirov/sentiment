import spacy
import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
headers = pd.read_csv("headers_only.csv")
rf_model = joblib.load('rf_model.joblib')
vectorizer = TfidfVectorizer(lowercase=False, max_features=10000, ngram_range=(1, 2))

@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')

def tokenize_and_lemmatize(text):
  """
  Tokenizes and lemmatizes the input text.

  Args:
      text: A string representing the input text.

  Returns:
      a list of lemmas
  """
  lemmas = []

  doc = nlp(text)

  for token in doc:
      if token.is_alpha and not token.is_stop:
          lemmas.append(token.lemma_)
          new_lemma_tokens = " ".join(lemmas)

  return new_lemma_tokens


def predict_condition(user_input):
  """
  Preprocesses user input, predicts skin condition using the SVM model,
  and returns the prediction result.

  Args:
      user_input: A string representing the user's description.

  Returns:
      A string with the predicted skin condition.
  """
  try:
      # Lowercase and remove punctuation
      user_input = user_input.lower()
      user_input = "".join([char for char in user_input if char.isalnum() or char.isspace()])
      
      # Tokenize and lemmatize
      lemmatized_text = tokenize_and_lemmatize(user_input)
      
      # Vectorize the text
      vectorized_input = vectorizer.fit_transform([lemmatized_text])
      vectorized_input = pd.DataFrame(vectorized_input.toarray(), columns=vectorizer.get_feature_names_out())
      vectorized_input = pd.DataFrame(vectorized_input)

      # Load missing columns from training data (if any)
      #missing_columns = set(headers.columns) - set(vectorized_input.columns)
      vectorized_input = vectorized_input.reindex(columns=headers.columns, fill_value=0)
      #vectorized_input = vectorized_input.assign(**{col: 0 for col in missing_columns})

      vectorized_input = vectorized_input[headers.columns]
      st.write(vectorized_input)
      # Predict using the model
      prediction = rf_model.predict(vectorized_input)[0]

      # Return the prediction result
      return prediction

  except Exception as e:
      # Handle potential errors during preprocessing or prediction
      error_message = f"An error occurred: {str(e)}"
      st.error(error_message)
      return None

# Streamlit app layout and functionality
st.title(":blue[Text Sentiment Analyzer] App")
user_input_text = st.text_area("Write your text here:", height=100)

if st.button("Predict"):
    prediction = predict_condition(user_input_text)
    if prediction:
        st.success(f"Predicted condition: {prediction}. \n")
    else:
        st.warning("Prediction failed. Please try again and/or consult with developers.")

# Display model information (optional)
st.header("Model Information")
st.header("This app is powered by a trained Random Forest model.", divider = 'rainbow')

st.write("- Text to be analyzed sentiment-wise")
st.write("- Data Science Club Group One")
st.write("- IG: @american_corner_petropavlovsk")
st.write("- Club host: Askhat Aubakirov")
st.write("- Publication Date: September 17th 2024")