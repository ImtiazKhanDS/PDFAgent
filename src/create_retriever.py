from typing import List
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def get_retriever(model_name:str, docs:List):
    """Returns the retriever for a given model and a list of docs"""
    splits = text_splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=HuggingFaceEmbeddings(model_name=model_name))
    retriever = vectorstore.as_retriever()
    return retriever