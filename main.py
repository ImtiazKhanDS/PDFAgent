from langchain_core.prompts import ChatPromptTemplate

from src.chains import create_stuff_documents_chain, create_retrieval_chain
from src.load_llm import load_llm
from src.create_retriever import get_retriever
from src.pdf_parser import get_documents
from src.prompts import system_prompt
from fastapi import FastAPI
import uvicorn


import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()



class PDFAgent:
    def __init__(self):
        self.llm_model_name = os.environ.get('LLM_MODEL_NAME')
        self.embedding_model_name = os.environ.get('EMBEDDING_MODEL_NAME')
        self.hugging_face_token = os.environ.get('HUGGING_FACE_HUB_API_TOKEN')
        self.max_tokens = os.environ.get('MAX_TOKENS', 512)
        self.file_path = os.environ.get('FILE_PATH')
        self.llm = load_llm(self.llm_model_name, self.max_tokens)
        self.docs = get_documents(self.file_path)
        self.retriever = get_retriever(self.embedding_model_name, self.docs)
        self.executor = self.create_chain()

    def create_chain(self):
        prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", system_prompt),
                                ("human", "{input}"),
                            ]
                        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        return rag_chain
    
    async def get_answer(self, messages:list):
        try:
            results = self.executor.invoke({"input": messages[-1]['content']})
            messages.append(results['answer'])
            return messages
        except Exception as e:
            messages.append(f"The answer to question could not be retrieved at the moment due to {e}")
            return messages
    

pdfAgent = PDFAgent()
@app.get("/")
def get_answers(question:str):
    res = pdfAgent.get_answer(question)
    return {"question":question, "answer":  res['answer']}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
       
        






