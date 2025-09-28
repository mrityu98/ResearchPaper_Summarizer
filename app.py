import streamlit as st
import tempfile
from rag_pipeline import load_pdf, create_retriever, answer_query, splitted_docs, create_embeddings

st.title("📄 Research Paper Summarizer")

uploaded_file = st.file_uploader("Upload a Research Paper PDF", type=["pdf"])
query = st.text_input("Ask a question about the paper")

if uploaded_file and query:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    docs = load_pdf(tmp_path)
    new_docs = splitted_docs(docs)
    db=create_embeddings(new_docs)
    retriever = create_retriever(docs,db)
    response = answer_query(query, retriever)
    st.write("### Response:")
    st.write(response)
