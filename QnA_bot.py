from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI( model="gemini-2.5-flash-lite")

st.title("AskBuddy AI QnA")
st.markdown(" My QnA Bot")

if "messages" not in st.session_state:
    st.session_state.messages= []

for messages in st.session_state.messages:
    role= messages["role"]
    content= messages["content"]
    st.chat_message(role).markdown(content)

    
query= st.chat_input("ask Anything")
if query:
    st.session_state.messages.append({ "role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res= llm.invoke(query)
    st.chat_message("ai").markdown(res.content)
    st.session_state.messages.append({ "role": "ai", "content": res.content })
   