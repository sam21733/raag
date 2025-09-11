import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Please set GOOGLE_API_KEY in your .env file or Hugging Face Space secrets.")
    st.stop()

st.set_page_config(page_title="RAG PDF QA", page_icon="📄")
st.title("📄 Retrieval-Augmented QA on PDF")

# File upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save to a temp file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Processing the PDF…")

    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    docs = text_splitter.split_documents(documents)

    # Embeddings
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    # Vector store
    vector_store = FAISS.from_documents(docs, embedding_function)
    retriever = vector_store.as_retriever()

    # LLM
    llm = ChatGoogleGenerativeAI(
        temperature=0, model="gemini-1.5-flash", api_key=api_key
    )

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )

    st.success("✅ PDF loaded and indexed")

    # Ask a question
    query = st.text_input("Ask a question about the document:")

    if query:
        with st.spinner("Thinking…"):
            response = qa_chain({"query": query})

        st.subheader("Answer")
        st.write(response["result"])

        with st.expander("Show Sources"):
            for doc in response["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source','unknown')}")
                st.write(doc.page_content[:500] + "…")
