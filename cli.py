import argparse
from src.utils import extract_text_from_pdf, generate_embeddings, extract_answer, check_confidence

def main(pdf_path, questions):
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)  # Define chunking logic
    embeddings = generate_embeddings(chunks)
    
    answers = []
    for question in questions:
        question_embedding = generate_embeddings([question])[0]
        relevant_chunks = get_most_relevant_chunks(embeddings, question_embedding)
        context = " ".join([chunks[i] for i in relevant_chunks])
        answer = extract_answer(question, context)
        if not check_confidence(answer):
            answer = "Data Not Available"
        answers.append({"question": question, "answer": answer})
    
    return answers
