# AgriBot
🌾 AgriBot: An intelligent agriculture QA assistant powered by RAG, FAISS, and Llama 3.

🌾 AgriBot — Agriculture QA Assistant
AgriBot is an intelligent agricultural assistant designed to answer farmers' and researchers' questions using knowledge from agriculture-specific datasets.
It combines retrieval-augmented generation (RAG) with FAISS vector search and a language model powered by Ollama.

✨ Features
Load and combine multiple agriculture QA datasets
Split text intelligently for better retrieval
Embed documents using HuggingFace sentence-transformers
Store and retrieve embeddings efficiently using FAISS
Generate accurate, context-based answers using Ollama's Llama 3.2 model
Interactive Command Line Interface (CLI)

📚 Datasets Used
KisanVaani/agriculture-qa-english-only
muhammad-atif-ali/agriculture-dataset-for-falcon-7b-instruct

🛠️ Tech Stack
Python 🐍
HuggingFace Datasets for loading QA data
LangChain for document processing
FAISS for efficient vector similarity search
HuggingFace Embeddings (all-MiniLM-L6-v2) for sentence embeddings
Ollama for local LLM inference (Llama 3.2 model)

🚀 How It Works
Load datasets from Hugging Face
Normalize documents into a consistent QA/instruction format
Split text into manageable chunks
Generate embeddings for each chunk
Store or load the FAISS vector index
Retrieve top-k relevant documents based on user queries
Generate answers using a language model with retrieved context

🔧 Installation
# Clone the repository
git clone https://github.com/yourusername/agribot.git
cd agribot

# Install required Python libraries
pip install torch ollama datasets langchain faiss-cpu sentence-transformers
Make sure you have Ollama installed and running locally to serve the llama3.2:latest model.

🖥️ Usage
python agribot.py

Example Interaction:
🌾 AgriBot is ready with extended knowledge! Ask me anything about agriculture.

Ask a question (or type 'exit' to quit): How do I protect my crops from pests?
🧠 Answer: [Detailed expert agricultural advice...]

📂 Project Structure
agribot/
│
├── agribot.py         # Main script
├── faiss_index/       # (Auto-created) Folder storing FAISS index
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies (optional)

🧠 Future Improvements
Add support for multilingual queries (e.g., Marathi, Hindi)
Fine-tune custom LLMs on agriculture domain
Improve retrieval ranking using semantic re-ranking models

🤝 Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request.
