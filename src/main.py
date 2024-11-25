# Import necessary libraries
import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

# Set the application title for the Streamlit interface
st.title("AI Medical Helper")

# Add a button for generating the knowledge base
# This button allows the user to create or regenerate the knowledge base from the dataset
btn = st.button("Generate Knowledge Base")
if btn:
    # Display a success message when the button is clicked
    st.success("Knowledge base successfully created!")

# Add a text input field where users can type their medical-related questions
question = st.text_input("Enter Your Medical Query:")

# Check if the user has entered a question
if question:
    # Get the QA chain for processing the query
    chain = get_qa_chain()

    # Generate the response using the QA chain for the entered question
    response = chain(question)
    
    # Display the answer section title
    st.subheader("Your AI-Powered Medical Insight:")

    # Display the generated response to the user's query
    st.write(response["result"])
