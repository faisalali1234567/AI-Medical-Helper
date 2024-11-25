# from dotenv import load_dotenv
# from langchain_google_genai import GoogleGenerativeAI
# from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA

# import os
# load_dotenv()


# llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.9)

# loader = CSVLoader(file_path="symptoms_disease_treatment1.csv", source_column="Symptoms")
# data = loader.load()


# embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb_file_path = "faiss_index"

# def create_vector_db():
#     loader = CSVLoader(file_path="symptoms_disease_treatment1.csv", source_column="Symptoms")
#     data = loader.load()

#     vectordb = FAISS.from_documents(documents=data,embedding=embeddings_model,)
#     vectordb.save_local(vectordb_file_path)

# def get_qa_chain():
#     vectordb = FAISS.load_local(vectordb_file_path,embeddings_model,allow_dangerous_deserialization=True)


#     retriever = vectordb.as_retriever(score_threshold = 0.9)

#     from langchain.prompts import PromptTemplate

#     prompt_template = """Given the following context and a question, generate an answer based on this context only.
#     In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
#     If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

#     CONTEXT: {context}

#     QUESTION: {question}"""

#     PROMPT = PromptTemplate(
#         template=prompt_template,input_variables=["context","question"]
#     )

#     chain = RetrievalQA.from_chain_type(llm=llm,
#                                     chain_type = "stuff",
#                                     retriever = retriever,
#                                     input_key = "query",
#                                     return_source_documents = True,
#                                     chain_type_kwargs = {"prompt":PROMPT}
#                                    )
#     return chain


# if __name__ =="__main__":
#     # create_vector_db()
    
#     chain = get_qa_chain()

#     print(chain("what is the medicine for fever and what dosage is one has to take "))

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import os
load_dotenv()


llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.5)

loader = CSVLoader(file_path="symptoms_disease_treatment1.csv", source_column="Symptoms")
data = loader.load()


embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path="symptoms_disease_treatment1.csv", source_column="Symptoms")
    data = loader.load()

    vectordb = FAISS.from_documents(documents=data,embedding=embeddings_model,)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path,embeddings_model,allow_dangerous_deserialization=True)


    retriever = vectordb.as_retriever(score_threshold = 0.9)

    from langchain.prompts import PromptTemplate

    prompt_template = """Given the following dataset context, answer questions specifically about Diseases, Symptoms, Medicine, Dosage, and Precautions:

    Dataset columns: Disease, Symptoms, Medicine, Dosage, Precautions.

    If the user asks for details of a disease, search for the exact match in the dataset. Provide only the available data for that disease. Format the output in the following structured manner:
    
    Medicine: <value>
    Dosage: <value>
    Precautions: <value>
    
    Ensure each field is displayed on a new line or provide a comma between each field displayed . If the disease is not found in the context, respond with "I don't know." Do not generate any information not present in the dataset.

    CONTEXT: {context}

    QUESTION: {question}"""





    PROMPT = PromptTemplate(
        template=prompt_template,input_variables=["context","question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type = "stuff",
                                    retriever = retriever,
                                    input_key = "query",
                                    return_source_documents = True,
                                    chain_type_kwargs = {"prompt":PROMPT}
                                   )
    return chain


if __name__ =="__main__":
    # create_vector_db()
    
    chain = get_qa_chain()

    print(chain("what is the medicine for fever and what dosage is one has to take "))