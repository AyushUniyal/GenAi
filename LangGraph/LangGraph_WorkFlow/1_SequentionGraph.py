from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from typing import TypedDict
from langchain_openai import ChatOpenAI
load_dotenv()

llm = ChatOpenAI() 
eval_ai = ChatAnthropic(model_name="claude-haiku-4-5-20251001")

class BlogState(TypedDict):
    title : str
    content : str
    outline : str
    evaluate : int

def create_outline(state : BlogState) -> BlogState:
    prompt=PromptTemplate(
        template="Create a detailed  ouline on the provided title \n title : {title}",
        input_variables=["title"]
    )
    
    title = state["title"]
    chain = prompt | llm
    state["outline"] = chain.invoke({"title":title}).content

    return state

def create_blog(state : BlogState) -> BlogState:

    title = state["title"]
    outline = state["outline"]

    prompt = PromptTemplate(
        template="Create a detailed blog in the following provided title and outline \n Title : {title} \n\n Outline : \n{outline}",
        input_variables=["title", "outline"]
    )

    chain = prompt | llm

    result = chain.invoke({"title":title, "outline":outline})

    state["content"] = result.content

    return state

def eval_blog(state : BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]
    content = state["content"]

    prompt = PromptTemplate(
        template="Based on below given title & Outline rate the content in out of 10. Only give number nothing else \n title:{title} \n \n outlinr : \n{outline} \n\n content : \n {content}",
        input_variables=["title","outline", "content"]
    )

    chain = prompt | eval_ai

    result = chain.invoke({"title": title, "outline": outline, "content" : content})
    state["evaluate"] = result.content
    return state

graph = StateGraph(BlogState)

graph.add_node("Create_Outline", create_outline)
graph.add_node("Create_Blog", create_blog)
graph.add_node("Evaluate_Blog", eval_blog)

graph.add_edge(START,"Create_Outline")
graph.add_edge("Create_Outline", "Create_Blog")
graph.add_edge("Create_Blog", "Evaluate_Blog")
graph.add_edge("Evaluate_Blog", END)


workflow = graph.compile()

print(workflow.get_graph().print_ascii())

initial_state = {"title" : "Raise of AI in INDIA"}

final_state = workflow.invoke(initial_state)

print(final_state)
