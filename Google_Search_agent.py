from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

search = GoogleSerperAPIWrapper()
llm= ChatGroq( model="openai/gpt-oss-20b")
memory= MemorySaver()

agent= create_agent(
    model=llm,
    tools=[ search.run],
    checkpointer= memory,
    system_prompt=" You are an agent and can search for any question on google"
)

while True:
    query= input("User:")
    if( query.lower()=="quit"):
        print( "Quiting chat")
        break

    res= agent.invoke({ "messages": [{"role":"user","content": query}]}, {"configurable": {"thread_id": "1"}})
    print( "AI:", res["messages"][-1].content)