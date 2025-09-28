from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def load_pdf(path):
    # return documents
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents

def splitted_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
    splitted_docs = splitter.split_documents(docs)
    return splitted_docs

def create_embeddings(docs):
    embeddings=OpenAIEmbeddings()
    db=Chroma.from_documents(docs,embeddings)
    return db

def create_retriever(docs,db):
    # return retriever
    retriever = db.as_retriever(strategy='similarity',search_kwargs={"k":10})
    return retriever

def answer_query(user_query, retriever):
    # return final LLM response
    promptTemplate="""
    You are an Data Science research paper specialist. Given the text input as {context} and a user query as {question}
    answer the question in a clear and concise manner.If asked about the model proposed in the work, focus more on the proposed model. If asked about the authors, they are mentioned at the top of the research papers.If you find the context to be insufficient to answer, then say that you don't
    know about it or the question is irrelevant to the paper."""
    prompt=PromptTemplate(template=promptTemplate,
                     input_variables=["context","question"])
    rel_docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([doc.page_content for doc in rel_docs])
    prompt=prompt.format(context=context,question=user_query)
    model = ChatOpenAI(model='gpt-3.5-turbo',temperature=0.3)
    response=model.predict(prompt)
    return response

