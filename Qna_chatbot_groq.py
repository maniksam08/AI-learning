from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm= ChatGroq( model="openai/gpt-oss-20b", streaming=True)

if "memory" not in st.session_state:
    st.session_state.memory= MemorySaver()
    st.session_state.history= [] 

agent= create_agent(
    model= llm,
    tools= [GoogleSerperAPIWrapper().run],
    checkpointer= st.session_state.memory,
    system_prompt=" You help the llm to provide google search"
)

st.subheader(" QnA faster than chatGPT")
query= st.chat_input("Ask Anything")

for message in st.session_state.history:
    role= message["role"]
    content= message["content"]
    st.chat_message(role).markdown(content)

if( query):
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content": query})

    response= agent.stream(
        { 
            "messages": [{"role":"user","content": query}]
        },
        {
            "configurable": {"thread_id": "1"}
        },
        stream_mode="messages"
    )

    container= st.chat_message("ai")
    with container:
        space= st.empty()
        message=""

        for res in response:
            message= message+ res[0].content
            space.write(message)
        
        st.session_state.history.append({"role":"ai","content": message})
