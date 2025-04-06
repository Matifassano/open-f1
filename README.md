# 🧠 In-Memory RAG with LangChain + Web Scraping
This project is a lightweight Retrieval-Augmented Generation (RAG) system that allows you to ask questions about scraped F1 websites contents. It scrapes one or more URLs, generates embeddings, and answers questions using that specific data — all stored in-memory with no persistent database.


## 🚀 Technologies Used
- 🦜 LangChain
- 🧠 OpenAI
- 🔍 ChromaDB
- 🧩 PromptTemplate


## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Matifassano/open-f1.git
   cd open-f1

2. Install dependencies
    ```bash
    pip install -r requirements.txt

3. Add your OpenAI key and F1 Urls to a `.env` file:
   OPENAI_API_KEY=your_openai_api_key
   URLS=https://example.com,https://another-url.com

4. Run the script
    ```bash
    streamlit run main.py

🧪 How It Works
1. Web content is scraped from the URLs in the `.env` file.
2. The text is split into semantic chunks.
3. Embeddings are generated with OpenAI and stored in-memory.
4. When a user asks a question, the system retrieves the most relevant chunks.
5. The LLM answers *only* using the retrieved context.