import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import os

# Set up environment variable for Google GenerativeAI API key
os.environ["GOOGLE_GENERATIVEAI_API_KEY"] = "AIzaSyBaoWpue25w75DSU71oyXQDJ6LfVhKJQfs"

st.title('CSV Question and Answer ChatBot')

# Allow user to upload CSV file
csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")

# Function to save uploaded file to folder
if csv_file_uploaded is not None:
    def save_file_to_folder(uploadedFile):
        save_folder = 'content'
        save_path = Path(save_folder, uploadedFile.name)
        st.write(f'Saving file to: {save_path}')  # Debugging output
        if os.path.exists(save_path):
            with open(save_path, mode='wb') as w:
                w.write(uploadedFile.getvalue())
            st.success(f'File {uploadedFile.name} is successfully saved!')
        else:
            st.error(f'Error: File {uploadedFile.name} does not exist at path: {save_path}')

    save_file_to_folder(csv_file_uploaded)

    # Load CSV file using CSVLoader
    loader = CSVLoader(file_path=os.path.join('content/', csv_file_uploaded.name))

    # Create an index using the loaded documents
    index_creator = VectorstoreIndexCreator()

    # Assuming the VectorstoreIndexCreator takes no OpenAI dependencies
    docsearch = index_creator.from_loaders([loader])

    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=GoogleGenerativeAI(), chain_type="stuff",
                                        retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    # Chatbot interface title
    st.title("Chat with your CSV Data")

    # Storing the chat history
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    # Function to generate response to user input
    def generate_response(user_query):
        response = chain({"question": user_query})
        return response['result']

    # Function to get user input
    def get_text():
        input_text = st.text_input("You: ", "Ask a question from your document?", key="input")
        return input_text

    user_input = get_text()

    if user_input:
        output = generate_response(user_input)
        # Store the output and user input
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    # Display chat history
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            st.write("Bot:", st.session_state["generated"][i])
            st.write("You:", st.session_state['past'][i])
