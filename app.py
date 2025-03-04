import streamlit as st
from streamlit_option_menu import option_menu
import sys
sys.path.append("..")

from chain import get_response

st.set_page_config(layout='centered', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo-16k"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Chat Now", "Recent Chat 1", "Recent Chat 2", "Recent Chat 3"], 
        icons=['chat'],   
        default_index=0)

user_input = st.chat_input("Enter a prompt here")

if selected == 'Chat Now':
    _placeholder = st.empty()

    if len(st.session_state.chat_history) == 0:
        with _placeholder.container():
            st.markdown("<br>"*2, unsafe_allow_html=True)
            c1, c2 = st.columns((0.90, 0.10))
            with c1:
                st.header('Hello!')
                st.header('How can I help you?')

            c4, c5, c6, c7 = st.columns(4)
            with c4:
                q1 = "Any good news about Philippine companies lately?"
                if st.button(q1):
                    user_input = q1

            with c5:
                q2 = "With the recent news, are there any bad moves done by company that may affect beginner investors like me?"
                if st.button(q2):
                    user_input = q2

            with c6:
                q3 = 'Give me news about Philippine banks.'
                if st.button(q3):
                    user_input = q3

            with c7:
                q4 = 'How can I become a millionaire overnight?'
                if st.button(q4):
                    user_input = q4

        if user_input:
            _placeholder.empty()
    else:
        _placeholder.empty()


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        placeholder = st.empty()

        with st.spinner("Thinking..."):
            response = get_response(user_input, st.session_state.chat_history)
            placeholder.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})