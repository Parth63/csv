import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Configure the Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')

# Function to generate response using the Gemini model
def get_gemini_response(input_prompt, csv_data, user_query):
    response = model.generate_content([input_prompt, csv_data, user_query])
    return response.text

# Load CSV file
def load_csv(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

# Initialize Streamlit app
st.set_page_config(page_title="CSV Chatbot")

st.header("CSV Chatbot")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load CSV file and display its contents
if uploaded_file is not None:
    df = load_csv(uploaded_file)
    if df is not None:
        st.success("CSV file uploaded successfully.")
        st.write(df.head())  # Display the first few rows of the CSV file
    else:
        st.error("Failed to load CSV file.")

# Get user input
query = st.text_input("Enter your query")

# Process user query
if st.button("Get Response"):
    if df is not None:
        try:
            # Define the input prompt for the Gemini model
            input_prompt = "You are an expert in understanding invoices."
            
            # Convert the CSV data to a format that can be processed by the Gemini model
            csv_data = df.to_dict(orient="list")
            
            # Use Gemini model to generate response
            response = get_gemini_response(input_prompt, csv_data, query)
            st.write("Response:", response)
        except Exception as e:
            st.error(f"Error processing query: {e}")
    else:
        st.error("No CSV file loaded.")
