# AI Medical Helper  

### Overview  
AI Medical Helper is a Generative AI-powered platform designed to assist users in finding medical information such as disease symptoms, recommended medicines, dosages, and precautions. By leveraging advanced tools like LangChain, FAISS, and Google PaLM, the platform provides accurate and reliable medical insights based on user queries.  

---

### Features  
- **User-Friendly Interface**: Built using Streamlit for an interactive and intuitive experience.  
- **Advanced Search**: Retrieves precise information about diseases, medicines, and dosages using vector search.  
- **Powered by Generative AI**: Utilizes LangChain and Google PaLM to process user queries.  
- **Customizable Dataset**: Processes medical data stored in CSV format for adaptability.  

---

### Tech Stack  
- **Frontend**: Streamlit  
- **Backend**:  
  - **LangChain**: For seamless integration of retrieval-based question answering.  
  - **FAISS**: For efficient vector search and indexing.  
  - **Google PaLM**: For natural language understanding and response generation.  
- **Data Handling**: HuggingFace Embeddings, CSVLoader  

---

### Installation and Setup  

1. Clone the repository:  
   ```bash  
   git clone https://github.com/faisalali1234567/AI-Medical-Helper.git  
   cd AI-Medical-Helper  
   ```  

2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. Add your environment variables in a `.env` file:  
   ```
   GOOGLE_API_KEY=<your_google_api_key>  
   ```  

4. Create the FAISS vector database:  
   ```bash  
   python langchain_helper.py  
   ```  

5. Run the application:  
   ```bash  
   streamlit run app.py  
   ```  

---

### Usage  

1. Launch the application on your browser.  
2. Click **Generate Knowledge Base** to initialize the dataset.  
3. Enter your query (e.g., "What medicine should I take for fever?").  
4. View the generated response, including the medicine, dosage, and precautions.  

---

### Demo Video  
Watch the platform in action:  
[Google Drive Demo Link](https://drive.google.com/file/d/1iQl5ViLoP26LIPjxSKU0r1CZdBQubP65/view?usp=sharing)  

---

### Team Members  
- **Faisal Ali**  
  Email: faisalstory123@gmail.com  
  Phone: 9535135109  

- **Shameer K**  
- **Adithya Atreya GK**  

---

### Future Scope  
- Expanding the dataset to include more diseases and treatments.  
- Enhancing the UI for better accessibility and features.  
- Integrating additional AI models for improved accuracy.  

---
