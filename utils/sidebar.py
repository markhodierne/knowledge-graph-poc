import streamlit as st
from langchain.memory import ConversationBufferMemory

def render_sidebar(unique_key: str = ""):
    if st.sidebar.button("Delete Chat History", key=f"delete_chat_history_{unique_key}"):
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="input"
        )
        st.session_state.messages = []