from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated

load_dotenv()
model = ChatOpenAI()
checkpoint = MemorySaver()

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