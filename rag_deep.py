import os
import openpyxl
import pandas as pd
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PDFPlumberLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


class RAG:

    def __init__(self):

        # ---------- Config ----------
        self.PDF_STORAGE_PATH = 'document_store/files/'
        self.FAISS_INDEX_PATH = 'vector_store/index'
        os.makedirs(self.PDF_STORAGE_PATH, exist_ok=True)
        os.makedirs("vector_store", exist_ok=True)
        self.EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
        self.LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")
    
        # Load FAISS or initialize
        if os.path.exists(self.FAISS_INDEX_PATH):
            self.VECTOR_STORE = FAISS.load_local(self.FAISS_INDEX_PATH, self.EMBEDDING_MODEL, allow_dangerous_deserialization=True)
        else:
            self.VECTOR_STORE = None


        self.PROMPT_TEMPLATE = """
            You are an expert research assistant. Use the provided context to answer the query. 
            If unsure, say you don't know. Be concise (max 3 sentences).

            Query: {user_query}
            Context: {document_context}
            Answer:
            """

    def uiStyle(self):

        # ---------- Style ----------
        st.markdown("""
            <style>
            .stApp { background-color: #0E1117; color: #FFFFFF; }
            .stChatInput input {
                background-color: #1E1E1E !important;
                color: #FFFFFF !important;
                border: 1px solid #3A3A3A !important;
            }
            .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
                background-color: #1E1E1E !important;
                border: 1px solid #3A3A3A !important;
                color: #E0E0E0 !important;
                border-radius: 10px; padding: 15px; margin: 10px 0;
            }
            .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
                background-color: #2A2A2A !important;
                border: 1px solid #404040 !important;
                color: #F0F0F0 !important;
                border-radius: 10px; padding: 15px; margin: 10px 0;
            }
            .stChatMessage .avatar {
                background-color: #00FFAA !important;
                color: #000000 !important;
            }
            .stFileUploader {
                background-color: #1E1E1E;
                border: 1px solid #3A3A3A;
                border-radius: 5px;
                padding: 15px;
            }
            h1, h2, h3 { color: #00FFAA !important; }
            </style>
        """, unsafe_allow_html=True)

    def save_file(self, uploaded_file):
        
       

        path = os.path.join(self.PDF_STORAGE_PATH, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return path

    def load_doc(self, path):
        ext = path.split('.')[-1].lower()
        if ext == "pdf":
            return PDFPlumberLoader(path).load()
        elif ext == "txt":
            return TextLoader(path).load()
        elif ext == "docx":
            return UnstructuredWordDocumentLoader(path).load()
        elif ext == "csv":
            return CSVLoader(path).load()
        elif ext == "xlsx":
            try:
                # Load all sheets
                xls = pd.read_excel(path, sheet_name=None)  # returns dict of {sheet_name: DataFrame}
                docs = []
                for sheet_name, df in xls.items():
                    for index, row in df.iterrows():
                        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                        # Add sheet name for traceability
                        page_content = f"(Sheet: {sheet_name})\n{row_text}"
                        #docs.append({"page_content": page_content})
                        docs.append(Document(page_content=page_content))
                return docs
            except Exception as e:
                st.warning(f"Could not read Excel file: {e}")
                return docs
        else:
            st.warning(f"Unsupported: {ext}")
            return []

    def split_docs(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    def index_docs(self, chunks):
        #global VECTOR_STORE
        if self.VECTOR_STORE:
            self.VECTOR_STORE.add_documents(chunks)
        else:
            self.VECTOR_STORE = FAISS.from_documents(chunks, self.EMBEDDING_MODEL)
        self.VECTOR_STORE.save_local(self.FAISS_INDEX_PATH)

    def search_docs(self, query):
        return self.VECTOR_STORE.similarity_search(query) if self.VECTOR_STORE else []

    def get_response(self, query, docs):
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        return (prompt | self.LANGUAGE_MODEL).invoke({"user_query": query, "document_context": context})

    def start(self):
        
        self.uiStyle()

        # ---------- Session State ----------
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "selected_query" not in st.session_state:
            st.session_state.selected_query = None
        if "document_loaded" not in st.session_state:
            st.session_state.document_loaded = False


        # ---------- Sidebar Chat History ----------
        st.sidebar.title("Chat History")

        if st.sidebar.button("Clear History"):
            st.session_state.chat_history.clear()

        for i, entry in enumerate(reversed(st.session_state.chat_history)):
            if st.sidebar.button(entry["user"], key=f"history_{i}"):
                st.session_state.selected_query = entry["user"]

        # ---------- Title & Upload ----------
        st.title("DeepMind UNITEL")
        st.markdown("----")

        uploaded_files = st.file_uploader(
            "Upload documents", type=["pdf", "docx", "txt", "csv", "xlsx"],
            accept_multiple_files=True
        )

        if uploaded_files and not st.session_state.document_loaded:
            all_chunks = []
            for file in uploaded_files:
                path = self.save_file(file)
                docs = self.load_doc(path)
                chunks = self.split_docs(docs)
                all_chunks.extend(chunks)
            self.index_docs(all_chunks)
            st.session_state.document_loaded = True
            st.success("✅ Documents indexed!")

        # ---------- Display Chat ----------
        for msg in st.session_state.chat_history:
            with st.chat_message("user"): st.write(msg["user"])
            with st.chat_message("assistant", avatar="🤖"): st.write(msg["ai"])

        # ---------- Chat Input ----------
        query = st.session_state.selected_query or st.chat_input("Ask something about your documents...")

        if query:
            with st.chat_message("user"): st.write(query)

            with st.spinner("Thinking..."):
                if self.VECTOR_STORE:
                    docs = self.search_docs(query)
                    self.response = self.get_response(query, docs)
                else:
                    response = self.LANGUAGE_MODEL.invoke(
                        f"You are an assistant. No documents are loaded. Answer this using general knowledge:\n\nQ: {query}\nA:"
                    )

            with st.chat_message("assistant", avatar="🤖"): st.write(self.response)

            st.session_state.chat_history.append({"user": query, "ai": self.response})
            st.session_state.selected_query = None


if __name__ == "__main__":
    unitel_AI = RAG()
    unitel_AI.start()

