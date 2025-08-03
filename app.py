import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ------------------------- PAGE CONFIG & STYLING -------------------------
# Set Streamlit page config
# Set page config
st.set_page_config(
    page_title="Multi-PDF Chatbot (Local Ollama)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
st.markdown("""
    <style>
    /* Background and layout */
    body, .main {
        background-color: #0D1117;
        color: #E6EDF3;
        font-family: 'Segoe UI', sans-serif;
    }

    .stApp {
        background-color: #0D1117;
    }

    header, footer, .css-zq5wmm, .css-1v0mbdj {
        background-color: #0D1117 !important;
        color: #E6EDF3 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #58A6FF !important;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #2EA043;
        transition: 0.3s ease;
    }

    /* File uploader */
    .css-1u5xkym {
        background-color: #161B22 !important;
        border-radius: 10px;
        padding: 15px;
    }

    /* Input field */
    .stTextInput > div > div > input {
        background-color: #161B22;
        color: white;
        border: 1px solid #30363D;
        border-radius: 8px;
    }

    /* Centered icons */
    .icon-links {
        text-align: center;
        margin-top: 30px;
    }

    .icon-links a {
        margin: 0 15px;
        font-size: 18px;
        color: #58A6FF;
        text-decoration: none;
    }

    .icon-links a:hover {
        color: #1F6FEB;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)



#nwo writing main fucntions

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, model_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks
#store chunks as embedding vectors in vector databse
def get_vector_store(text_chunk, model_name, api_key=None):
    if model_name == "Ollama":
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = FAISS.from_texts(text_chunk, embedding=embeddings) # ab yahan per vector data base bana raha jisme chunk bhe pass kar raha orembeddder bhe ke embdedd karke vector db me store karle
    vector_store.save_local("faiss_index")
    return vector_store
#creating a converstional chain usig the lang chain
def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    if model_name == "Ollama":
        prompt_template = """
        Answer The Question as Detailed As Possible from the provided
        context and make sure to provide all the detail
        with proper structure and the source of the context
        , if teh answer is not in the provided context ask that the ansswer in not available in the context,
        and ask if to do web search , and if yess then do the web search,but first alwasy deeply seacrh the context
        and answer from there \n\n\n
        Context:\n{context}?\n
        Question:\n{question}?\n
        Answer:
        """
        model = ChatOllama(model="llama3.2")
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

#now tracking user input
def user_input(user_question, model_name, api_key, pdf_docs, conversation_history):
    if pdf_docs is None:
        st.warning("Please upload PDF files before processing.")
        return
#now calling each function with the help of one another
    text_chunks = get_text_chunks(get_pdf_text(pdf_docs), model_name)#chunks banayega pehle text lega text wale function se pdf ko pass karate hue
    vector_store = get_vector_store(text_chunks, model_name)# phir ooper wale chunks ko vector store ke func me call karwayega jahan embeddings ka func call horah aor wahan yeh sare chunks embedd horker stre horahe data base vector mein

    user_question_output = ""
    response_output = ""

    if model_name == "Ollama":
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")# now it will search the embedded data in the stored faiss data to find relative data from user quetsion
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)## ab jo new_d me poora faiss me hamara embeddedvector db houga save ab usme similarity seach karenge question ke embedded vector se ke kia match horaha
        chain = get_conversational_chain(model_name, vectorstore=new_db)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)#true because we dont need any meta data or any thing else
        

        user_question_output = user_question
        response_output = response['output_text']
        pdf_names = [pdf.name for pdf in pdf_docs] if pdf_docs else []
        conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ", ".join(pdf_names)))

        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{user_question_output}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{response_output}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if len(conversation_history) == 1:
        conversation_history = []
    elif len(conversation_history) > 1:
        last_item = conversation_history[-1]
        conversation_history.remove(last_item)

    for question, answer, model_name, timestamp, pdf_name in reversed(conversation_history):
        st.markdown(
            f"""
            <div class="chat-message user">
                <div class="avatar">
                    <img src="https://i.ibb.co/CKpTnWr/user-icon-2048x2048-ihoxz4vq.png">
                </div>
                <div class="message">{question}</div>
            </div>
            <div class="chat-message bot">
                <div class="avatar">
                    <img src="https://i.ibb.co/wNmYHsx/langchain-logo.webp">
                </div>
                <div class="message">{answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if len(st.session_state.conversation_history) > 0:
        df = pd.DataFrame(st.session_state.conversation_history, columns=["Question", "Answer", "Model", "Timestamp", "PDF Name"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download conversation history as CSV file</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)
        st.markdown("To download the conversation, click the Download button on the left side at the bottom of the conversation.")
    st.snow()

# ------------------------- MAIN FUNCTION -------------------------
def main():
    st.markdown("<h1 style='color:#000000;'>📖 Multi-PDF Chatbot (Local Ollama)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#1e3c72;'>💬 Ask questions from multiple PDFs using your local LLM</h3>", unsafe_allow_html=True)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    model_name = st.sidebar.radio("Select the Model:", ("Ollama",))
    api_key = None

    with st.sidebar:
        st.title("Menu")
        col1, col2 = st.columns(2)
        reset_button = col2.button("Reset")
        clear_button = col1.button("Rerun")

        if reset_button:
            st.session_state.conversation_history = []
            st.session_state.user_question = None
            pdf_docs = None
        elif clear_button:
            if 'user_question' in st.session_state:
                st.warning("The previous query will be discarded.")
                st.session_state.user_question = ""
                if len(st.session_state.conversation_history) > 0:
                    st.session_state.conversation_history.pop()

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", 
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("..")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, st.session_state.conversation_history)
        st.session_state.user_question = ""

    # 🔗 Professional Connect Section
    st.markdown("""
        <hr>
        <div style="background-color:#f0f2f6; padding:20px; border-radius:12px;">
            <h3 style="color:#1e3c72;">📡 Connect with Me</h3>
            <p style="font-size:16px;">
                I'm <strong>Waniya Badar</strong> and I have always had a keen interest in Artificial Intelligence and Software Development.
                You can react me out through the links below:
            </p>
            <p style="font-size:16px;">
                🔗 <a href="https://www.linkedin.com/in/waniya-badar/" target="_blank" style="text-decoration:none; color:#0a66c2;"><b>LinkedIn</b></a> &nbsp; | &nbsp;
                💻 <a href="https://github.com/waniya-badar" target="_blank" style="text-decoration:none; color:#333;"><b>GitHub</b></a>
            </p>
        </div>
        <br>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

#to run: streamlit run app.py