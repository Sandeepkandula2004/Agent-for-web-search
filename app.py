import streamlit as st
from dotenv import load_dotenv
import os

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import (
    ArxivAPIWrapper,
    WikipediaAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
)
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# --- Initialize Tools ---
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200))
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200))
search = DuckDuckGoSearchRun(name="Search")

tools = [search, arxiv, wiki]

# --- Streamlit UI ---
st.set_page_config(page_title="Search Agent with History", layout="centered")
st.title("🤖 Agent using Tools + Conversation History")
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("🔑 GROQ API Key", type="password")

if not api_key:
    st.warning("Please enter your GROQ API key in the sidebar.")
    st.stop()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a chatbot!"}]

# Display past chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask me anything (e.g., What is machine learning?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(api_key=api_key, model="Gemma2-9b-It", streaming=True)

    search_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handling_parsing_errors = True,
        verbose=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except Exception as e:
            response = f"❌ Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
