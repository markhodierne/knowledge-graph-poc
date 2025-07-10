import streamlit as st

def write_message(role, content, save = True):
    """
    This is a helper function that saves a message to the
    session state and then writes a message to the UI
    """
    if save:
        st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.markdown(content)


