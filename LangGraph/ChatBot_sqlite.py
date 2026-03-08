from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated
import sqlite3

load_dotenv()
model = ChatOpenAI()
conn = sqlite3.connect(database="Chatbot", check_same_thread=False)




checkpoint = SqliteSaver(conn=conn)

class chatState(TypedDict):
    message : Annotated[list[BaseMessage], add_messages]


def chat(state : chatState) -> chatState:

    response = model.invoke(state["message"])

    return {"message":[response]}

graph = StateGraph(chatState)
graph.add_node("chatBot", chat)

graph.add_edge(START, "chatBot")
graph.add_edge("chatBot", END)

workflow = graph.compile(checkpointer=checkpoint)

def retrieve_chat():
    all_thread = set()
    for chk in checkpoint.list(None):
        all_thread.add(chk.config["configurable"]["thread_id"])

    return list(all_thread)


def main():
    thread_id= "1"
    while True:
        user = input("User : ")
        if user.lower().strip() in ["quit", "exit", "bye"]:
            break
        else :
            config = {"configurable" : {"thread_id":thread_id}}
            response = workflow.invoke({"message":[HumanMessage(content=user)]}, config=config)
            print("AI : ",response["message"][-1].content)

if __name__ == "__main__":
    main()