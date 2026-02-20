__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import tempfile
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
from sqlalchemy import create_engine, text

# --- PATH SETUP ---
current_dir = os.getcwd() # Should be /home/user/app in Docker
load_dir = os.path.join(current_dir, "src", "load")
sys.path.append(load_dir)

# Import your agent creator and configurations
try:
    from mshauri_demo import create_mshauri_agent, DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL
except ImportError as e:
    st.error(f"Critical Error: Could not import mshauri_demo. Paths checked: {sys.path}. Details: {e}")
    st.stop()

# --- GLOBALS FOR DB PATHS ---
# SQLAlchemy requires a URI starting with sqlite:///
sql_path = f"sqlite:///{os.path.join(current_dir, 'mshauri_fedha_v6.db')}"
vector_path = os.path.join(current_dir, "mshauri_fedha_chroma_db")

# --- SESSION MANAGEMENT & CLEANUP ---
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temp_tables" not in st.session_state:
        st.session_state.temp_tables = []
    if "temp_doc_ids" not in st.session_state:
        st.session_state.temp_doc_ids = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = set()

def cleanup_ephemeral_data():
    """Drops temporary SQL tables and ChromaDB chunks if consent was not given."""
    # 1. Clean up SQL Tables
    if st.session_state.temp_tables:
        engine = create_engine(sql_path)
        with engine.connect() as conn:
            for table in st.session_state.temp_tables:
                try:
                    if not table.replace("_", "").isalnum():
                        print(f"Skipping invalid table name: {table}")
                        continue
                    conn.execute(text(f"DROP TABLE IF EXISTS \"{table}\""))
                    conn.commit()
                except Exception as e:
                    print(f"SQL Cleanup error: {e}")
        st.session_state.temp_tables = []

    # 2. Clean up Vector DB Documents
    if st.session_state.temp_doc_ids:
        try:
            embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=DEFAULT_OLLAMA_URL)
            vectorstore = Chroma(persist_directory=vector_path, embedding_function=embeddings)
            vectorstore.delete(ids=st.session_state.temp_doc_ids)
        except Exception as e:
            print(f"Vector Cleanup error: {e}")
        st.session_state.temp_doc_ids = []

# --- FAST EXTRACTION PIPELINE ---

def process_uploaded_file(uploaded_file, consent):
    """Handles fast extraction based on file type and applies consent rules."""
    file_name = uploaded_file.name
    
    # Skip if already processed in this session
    if file_name in st.session_state.uploaded_files:
        return
        
    st.session_state.uploaded_files.add(file_name)

    try:
        # ==========================================
        # PATH 1: TABULAR DATA -> SQL DATABASE
        # ==========================================
        if file_name.endswith(('.csv', '.xlsx', '.xls')):
            # Read the file based on extension
            if file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file) # Requires 'openpyxl'
                
            # Sanitize table name (e.g., "Q3 Budget.xlsx" -> "user_upload_q3_budget")
            safe_name = file_name.rsplit('.', 1)[0].replace(" ", "_").lower()
            safe_table_name = f"user_upload_{safe_name}"
            
            engine = create_engine(sql_path)
            df.to_sql(safe_table_name, con=engine, if_exists='replace', index=False)
            
            if not consent:
                st.session_state.temp_tables.append(safe_table_name)
                st.sidebar.success(f"Spreadsheet loaded ephemerally! Agent can query `{safe_table_name}`.")
            else:
                st.sidebar.success(f"Spreadsheet saved persistently as `{safe_table_name}`.")

        # ==========================================
        # PATH 2: UNSTRUCTURED DATA -> VECTOR DB
        # ==========================================
        elif file_name.endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # 1. Fast Markdown Extraction (Preserves Tables!)
            md_text = pymupdf4llm.to_markdown(tmp_path)

            # 2. Chunk the Markdown intelligently
            # Markdown splitter ensures tables aren't cut in half
            splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(md_text)

            # 3. Convert to LangChain Document objects
            docs = [Document(page_content=chunk, metadata={"source": file_name}) for chunk in chunks]

            # 4. Embed and Store
            embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=DEFAULT_OLLAMA_URL)
            vectorstore = Chroma(persist_directory=vector_path, embedding_function=embeddings)

            doc_ids = vectorstore.add_documents(docs)

            if not consent:
                st.session_state.temp_doc_ids.extend(doc_ids)
                st.sidebar.success("PDF loaded securely for this session only.")
            else:
                st.sidebar.success("PDF saved to persistent database.")

            os.unlink(tmp_path)

        # ==========================================
        # PATH 3: DOCUMENT FILES -> VECTOR DB
        # ==========================================
        elif file_name.endswith(('.docx', '.txt', '.md')):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Load document based on type
            if file_name.endswith('.docx'):
                loader = Docx2txtLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path)

            raw_docs = loader.load()
            raw_text = "\n".join([doc.page_content for doc in raw_docs])

            # Chunk the text
            splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(raw_text)

            # Convert to LangChain Document objects
            docs = [Document(page_content=chunk, metadata={"source": file_name}) for chunk in chunks]

            # Embed and Store
            embeddings = OllamaEmbeddings(model=DEFAULT_EMBED_MODEL, base_url=DEFAULT_OLLAMA_URL)
            vectorstore = Chroma(persist_directory=vector_path, embedding_function=embeddings)

            doc_ids = vectorstore.add_documents(docs)

            if not consent:
                st.session_state.temp_doc_ids.extend(doc_ids)
                st.sidebar.success(f"{os.path.splitext(file_name)[1].upper()} file loaded securely for this session only.")
            else:
                st.sidebar.success(f"{os.path.splitext(file_name)[1].upper()} file saved to persistent database.")

            os.unlink(tmp_path)

        else:
            st.sidebar.error("Unsupported file type.")

    except Exception as e:
        st.sidebar.error(f"Error processing {file_name}: {e}")


# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(page_title="Mshauri Fedha", page_icon="🦁")

init_session_state()

st.title("🦁 Mshauri Fedha")
st.markdown("### AI Financial Advisor for Kenya")

# --- SIDEBAR: UPLOAD & CONSENT ---
st.sidebar.header("📁 Data Upload & Security")
st.sidebar.markdown("Upload your own financial reports or datasets (Max 10 files).")
st.sidebar.info("💡 **Tip:** For best results, upload raw data (like financial ledgers) as **CSV/Excel**. Upload narrative reports as **PDFs**.")

consent = st.sidebar.checkbox(
    "I consent to securely storing this document in the database for future reference.", 
    value=False,
    help="If unchecked, your data is treated as ephemeral. It will be deleted instantly when you clear the chat."
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, CSV, XLSX, DOCX, TXT", 
    type=['pdf', 'csv', 'xlsx', 'xls', 'docx', 'txt', 'md'],
    accept_multiple_files=True
)

if uploaded_files:
    # Enforce the 10 file limit
    if len(uploaded_files) > 10:
        st.sidebar.error("⚠️ Please upload a maximum of 10 files at a time.")
    else:
        with st.sidebar:
            with st.spinner(f"Processing {len(uploaded_files)} file(s)..."):
                # Loop through all uploaded files
                for uploaded_file in uploaded_files:
                    process_uploaded_file(uploaded_file, consent)

st.sidebar.markdown("---")

# --- AGENT INITIALIZATION ---
if "agent" not in st.session_state:
    with st.spinner("Initializing Mshauri Brain (Loading Models & Data)..."):
        # Check if baseline data exists (Debugging for Space deployment)
        real_db_path = os.path.join(current_dir, "mshauri_fedha_v6.db")
        if not os.path.exists(real_db_path):
            st.warning(f"Database not found at {real_db_path}. Using empty state.")
            
        try:
            st.session_state.agent = create_mshauri_agent(
                sql_db_path=sql_path, 
                vector_db_path=vector_path
            )
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")

# --- CHAT UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about inflation, your uploaded data, or economic trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                if st.session_state.agent:
                    response = st.session_state.agent.invoke({"input": prompt})
                    output_text = response.get("output", "Error generating response.")
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                else:
                    st.error("Agent failed to initialize. Please refresh the page.")
            except Exception as e:
                st.error(f"An error occurred: {e}")