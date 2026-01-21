__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

# --- PATH SETUP ---
# Since we run this from the root via Docker, we need to help python find 'mshauri_demo'
# We add 'src/load' to the system path.
current_dir = os.getcwd() # Should be /home/user/app in Docker
load_dir = os.path.join(current_dir, "src", "load")
sys.path.append(load_dir)

# Import agent creator
try:
    from mshauri_demo import create_mshauri_agent
except ImportError as e:
    st.error(f"Critical Error: Could not import mshauri_demo. Paths checked: {sys.path}. Details: {e}")
    st.stop()

st.set_page_config(page_title="Mshauri Fedha", page_icon="🦁")

st.title("🦁 Mshauri Fedha")
st.markdown("### AI Financial Advisor for Kenya")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    with st.spinner("Initializing Mshauri Brain (Loading Models & Data)..."):
        # The Dockerfile downloads the data to the root folder (/home/user/app)
        sql_path = os.path.join(current_dir, "mshauri_fedha_v6.db")
        vector_path = os.path.join(current_dir, "mshauri_fedha_chroma_db")
        
        # Check if data exists (Debugging for Space deployment)
        if not os.path.exists(sql_path):
            st.error(f"❌ Database not found at {sql_path}. Did the clone fail?")
            st.stop()
            
        try:
            # Force the 7b model here to ensure CPU compatibility
            st.session_state.agent = create_mshauri_agent(
                sql_db_path=sql_path, 
                vector_db_path=vector_path,
                llm_model="qwen2.5:7b" # Force 2.5 7 B which can use CPU basic
            )
            st.success("System Ready!")
        except Exception as e:
            st.error(f"Failed to initialize agent: {e}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Ask about inflation, exchange rates, or economic trends..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                response = st.session_state.agent.invoke({"input": prompt})
                output_text = response.get("output", "Error generating response.")
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"An error occurred: {e}")