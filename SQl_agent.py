from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st

db= SQLDatabase.from_uri("sqlite:///tasks.db")

db.run("""
    create table if not exists tasks (
       ID integer primary key autoincrement,
       Title text not null,
       Description text,
       Status text check( Status in ('pending','in progress','completed')) default 'pending',
       Created_At timestamp default current_timestamp
       );
""")

model= ChatGroq(model="openai/gpt-oss-20b")
toolkit= SQLDatabaseToolkit(db=db, llm=model)
tools= toolkit.get_tools()


system_prompt="""
    You are a task management assistant that interacts with a SQL database containing a 'tasks' table. 

TASK RULES:
1. Limit SELECT queries to 11 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 11
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

@st.cache_resource
def get_agent():
    agent= create_agent(
    model= model,
    tools=tools,
    checkpointer= InMemorySaver(),
    system_prompt= system_prompt
    )
    return agent

agent= get_agent()

st.header("TaskAgent")

if "messages" not in st.session_state:
    st.session_state.messages= []

query= st.chat_input("ask anything")

for chunks in st.session_state.messages:
    role= chunks["role"]
    content= chunks["content"]
    st.chat_message(role).markdown(content)

if query:
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({ "role":"user", "content": query})

    with st.chat_message("ai"):
        with st.spinner("processing..."):
            res= agent.invoke(
                {"messages": [{"role":"user", "content": query}]},
                {"configurable":{"thread_id":"1"}}
            )
            print(res)
            result= res["messages"][-1].content
            st.session_state.messages.append({ "role":"ai", "content": result})
            st.markdown(result)
    
