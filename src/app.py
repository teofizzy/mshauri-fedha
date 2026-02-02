__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os

# --- PATH SETUP ---
current_dir = os.getcwd() # Should be /home/user/app in Docker
load_dir = os.path.join(current_dir, "src", "load")
sys.path.append(load_dir)

# Import your agent creator
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
        # SQLAlchemy requires a URI starting with sqlite:///
        # We use 4 slashes (sqlite:////) because it is an absolute path on Linux
        sql_path = f"sqlite:///{os.path.join(current_dir, 'mshauri_fedha_v6.db')}"
        vector_path = os.path.join(current_dir, "mshauri_fedha_chroma_db")
        
        # Check if data exists (Debugging for Space deployment)
        real_db_path = os.path.join(current_dir, "mshauri_fedha_v6.db")
        if not os.path.exists(real_db_path):
            st.error(f"Database not found at {real_db_path}. Did the clone fail?")
            st.stop()
            
        try:
            # mshauri_demo.py to intelligently pick the API or Local model.
            st.session_state.agent = create_mshauri_agent(
                sql_db_path=sql_path, 
                vector_db_path=vector_path
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
                if st.session_state.agent:
                    response = st.session_state.agent.invoke({"input": prompt})
                    output_text = response.get("output", "Error generating response.")
                    st.markdown(output_text)
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                else:
                    st.error("Agent failed to initialize. Please refresh the page.")
            except Exception as e:
                st.error(f"An error occurred: {e}")