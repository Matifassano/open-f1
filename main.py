import streamlit as st
import os
import shelve
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

st.title("OpenF1 Chatbot")
st.write("This is a chatbot about F1 news and information")
st.write("My name is OpenF1 and I'm here to help you with your F1 questions.")
st.markdown("---")

user_logo = "😎"
bot_logo = "🤖"
urls = os.getenv("URLS").split(",")

loader = WebBaseLoader(urls)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=80)
split_docs = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = Chroma.from_documents(
    split_docs,
    embeddings,
    collection_name="f1-chat",
    persist_directory=None
)

template = """You are a chatbot name "OpenF1" who have only knowledge about Formula 1 
                and only respond based on the knowledge provided. 
                If you don't know the answer, don't make it up and tell 
                the user that they can search on official Formula 1 pages.

{context}

Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

if "messages" not in st.session_state:
    def load_chat_history():
        with shelve.open("chat_history") as db:
            return db.get("messages", [])
    st.session_state.messages = load_chat_history()

def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

with st.sidebar:
    st.title("Sidebar")
    st.write("Here you can delete the chat history")
    if st.button("Delete Chat"):
        st.session_state.messages = []
        save_chat_history([])

for message in st.session_state.messages:
    avatar = user_logo if message["role"] == "user" else bot_logo
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_logo):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=bot_logo):
        message_placeholder = st.empty()
        full_response = qa_chain.run(prompt)
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_chat_history(st.session_state.messages)
