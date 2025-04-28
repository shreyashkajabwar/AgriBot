import os
import torch
import ollama
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# === Load Datasets ===
dataset_1 = load_dataset("KisanVaani/agriculture-qa-english-only", split="train")
dataset_2 = load_dataset("muhammad-atif-ali/agriculture-dataset-for-falcon-7b-instruct", split="train")

# === Normalize and Combine Documents ===
documents = []

# Dataset 1: QA format
for row in dataset_1:
    qa_text = f"Question: {row['question']}\nAnswer: {row['answers']}"
    documents.append(Document(page_content=qa_text, metadata={"source": "kisan-vaani"}))

# Dataset 2: Falcon-style instruct format
for row in dataset_2:
    instruct_text = f"Instruction: {row['question']}\nResponse: {row['answers']}"
    documents.append(Document(page_content=instruct_text, metadata={"source": "falcon-instruct"}))

# === Split Text into Chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)

# === Load Embedding Model ===
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Create or Load FAISS Vector Store ===
if not os.path.exists("faiss_index"):
    vector_store = FAISS.from_documents(split_docs, embedding_model)
    vector_store.save_local("faiss_index")
else:
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

# === Function to Retrieve and Generate Answers ===
def get_answer(question: str):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "Sorry, I couldn't find any relevant information to answer your question."

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are an expert agricultural assistant. Based on the following context, answer the user's question clearly and concisely.

Context:
{context}

User Question:
{question}

Answer:"""

    response = ollama.chat(model="llama3.2:latest", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

# === Interactive CLI ===
if __name__ == "__main__":
    print("🌾 AgriBot is ready with extended knowledge! Ask me anything about agriculture.")
    while True:
        user_query = input("\nAsk a question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Goodbye! 👋")
            break
        answer = get_answer(user_query)
        print("\n🧠 Answer:", answer)
