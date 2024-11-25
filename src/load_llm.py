from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def load_llm(model_name:str, max_new_tokens:int,do_sample:bool=False):
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.03,
    )

    chat_model = ChatHuggingFace(llm=llm)
    return chat_model