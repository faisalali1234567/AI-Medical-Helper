# import streamlit  as st

# from langchain_helper import create_vector_db,get_qa_chain

# st.title("medical information")
# btn = st.button("Create Knowledgebase")
# if btn:
#     pass

# question = st.text_input("QUESTION: ")

# if question:
#     chain = get_qa_chain()
#     response = chain(question)

#     st.header("Answer: ")

#     st.write(response["result"])

import streamlit as st
from langchain_helper import create_vector_db, get_qa_chain

# Set the application title
st.title("AI Medical Helper")

# Add a button to create the knowledge base
btn = st.button("Generate Knowledge Base")
if btn:
    st.success("Knowledge base successfully created!")

# Input field for user to enter their medical query
question = st.text_input("Enter Your Medical Query:")

# Display the answer when a question is provided
if question:
    chain = get_qa_chain()
    response = chain(question)
    
    st.subheader("Your AI-Powered Medical Insight:")
    st.write(response["result"])
