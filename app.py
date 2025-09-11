# app.py
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.set_page_config(page_title="RAG App", page_icon="📘")
st.title("📘 Retrieval-Augmented Generation (RAG) App")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Vector DB
    vector_store = FAISS.from_documents(docs, embedding_function)
    retriever = vector_store.as_retriever()

    # LLM
    llm = ChatGoogleGenerativeAI(
        temperature=0, model="gemini-1.5-flash", api_key=api_key
    )

    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    st.success("✅ Document uploaded and processed!")

    query = st.text_input("Ask a question about the document:")
    if query:
        response = qa_chain({"query": query})

        st.subheader("Answer")
        st.write(response["result"])

        with st.expander("Show Sources"):
            for doc in response["source_documents"]:
                st.markdown(f"**Page Source:** {doc.metadata['source']}")
                st.write(doc.page_content[:500] + "...")
