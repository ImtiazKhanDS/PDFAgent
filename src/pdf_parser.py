from langchain_community.document_loaders import PyPDFLoader

def get_documents(file_path : str):
    """Take the path of the file and return a list of documents"""
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs