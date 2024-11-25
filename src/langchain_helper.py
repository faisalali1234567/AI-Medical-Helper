# Import necessary libraries
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Load environment variables from .env file
load_dotenv()

# Initialize the Google Generative AI LLM with specific parameters
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5)

# Load data from the CSV file; the 'Symptoms' column is used as the source column
loader = CSVLoader(file_path="symptoms_disease_treatment1.csv", source_column="Symptoms")
data = loader.load()

# Define the embedding model for vector creation
embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the file path where the FAISS vector database will be saved
vectordb_file_path = "faiss_index"

# Function to create a vector database from the CSV data
def create_vector_db():
    loader = CSVLoader(file_path="symptoms_disease_treatment1.csv", source_column="Symptoms")
    data = loader.load()

    # Use the embedding model to create a FAISS vector store and save it locally
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings_model)
    vectordb.save_local(vectordb_file_path)

# Function to retrieve the QA chain for processing queries
def get_qa_chain():
    # Load the FAISS vector database locally
    vectordb = FAISS.load_local(vectordb_file_path, embeddings_model, allow_dangerous_deserialization=True)

    # Create a retriever with a score threshold for matching queries
    retriever = vectordb.as_retriever(score_threshold=0.9)

    # Define a prompt template for generating responses based on the dataset
    prompt_template = """Given the following dataset context, answer questions specifically about Diseases, Symptoms, Medicine, Dosage, and Precautions:

    Dataset columns: Disease, Symptoms, Medicine, Dosage, Precautions.

    If the user asks for details of a disease, search for the exact match in the dataset. Provide only the available data for that disease. Format the output in the following structured manner:
    
    Medicine: <value>
    Dosage: <value>
    Precautions: <value>
    
    Ensure each field is displayed on a new line or provide a comma between each field displayed. If the disease is not found in the context, respond with "I don't know." Do not generate any information not present in the dataset.

    CONTEXT: {context}

    QUESTION: {question}"""

    # Create a prompt using the template
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Build the retrieval-based QA chain using the LLM and retriever
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

# Main execution block
if __name__ == "__main__":
    # Uncomment this line to create the vector database before using the QA chain
    # create_vector_db()

    # Generate the QA chain
    chain = get_qa_chain()

    # Test the chain with a sample query
    print(chain("what is the medicine for fever and what dosage is one has to take "))
