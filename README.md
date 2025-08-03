# PAGE---PDF-Analysis-Guidance-Engine
PAGE - A powerful, local AI assistant that can read, analyze, and answer questions from any PDF document, all privately on your machine. Built with **LangChain**, **FAISS**, **Ollama**, and **Streamlit**, it offers a seamless experience to chat with your documents in natural language.

### Features:

- Upload one or multiple PDF files
- Ask open-ended or specific questions from the PDFs
- Fast response powered by a locally running LLM (like Gemma or Llama2 via Ollama)
- Secure: runs completely locally (no API calls to external servers)
- Embedded vector search via **FAISS**
- Clean and minimal Streamlit interface

### Tech Stack:

| Layer           | Tool/Library     | Purpose                            |
|----------------|------------------|------------------------------------|
| 🧠 Language Model | [Ollama](https://ollama.com)       | Run local LLMs like LLaMA, Gemma etc |
| 🦜🔗 LangChain     | `langchain`      | Chains, prompts, and agent logic   |
| 🧾 PDF Parsing   | `PyPDF2`         | Extracts raw text from PDFs        |
| 📂 Vector Store | `FAISS`          | Semantic search over PDF chunks    |
| 🎛 UI Layer     | `Streamlit`      | Interactive web UI                 |
| 🔢 Embeddings   | `ollama-embeddings` | Converts text into vectors         |


### Running the App:
streamlit run main.py

### How It Works:
1. PDF Upload
You upload one or more PDF documents.

2. Text Extraction & Chunking
The app extracts and splits the text using LangChain's CharacterTextSplitter.

3. Embedding
Each chunk is converted into a vector using Ollama's embedding model.

4. Vector Indexing
FAISS stores the vectorized chunks for fast similarity search.

5. Question Answering
Your query is embedded, compared against stored vectors, and the relevant chunks are passed to the LLM (via LangChain) to generate a response.

