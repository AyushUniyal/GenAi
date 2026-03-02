from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from typing import TypedDict, Annotated
import operator
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate

class structure(BaseModel):
    feedback : str = Field(description="Detailed Feedback of the essay")
    score : int = Field(description="Score out of 10", le=10, ge=0)

load_dotenv()

model = ChatOpenAI()
an_model = ChatAnthropic(model_name="claude-haiku-4-5-20251001")
structured_model = an_model.with_structured_output(structure)

parser = StrOutputParser()
class EssayEval(TypedDict):
    language_feedback : str
    analysis_feedback : str
    clarity_feedback : str
    overall_feedback : str
    individual_score : Annotated[list[int], operator.add]
    avg_score : int
    outline : str
    Essay : str
    topic :str

def create_outline(state : EssayEval) -> EssayEval:
    template = PromptTemplate(
        template= "Create a detailed outline on given below given topic \n Topic : {topic}",
        input_variables=["topic"]
    )

    chain = template | model
    result = chain.invoke({"topic" : state["topic"]})
    return {"outline" : result.content}

def create_Essay(state : EssayEval) -> EssayEval:
    template = PromptTemplate(
        template = "Craete a detailed Essay on below give Topic and outline \n Topic : {topic} \n\n Outline : \n {outline}",
        input_variables=["topic","outline"]
    )

    chain = template | model
    result = chain.invoke({"topic" : state["topic"], "outline" : state["outline"]})

    return {"Essay": result.content}
    
def language_feedback(state : EssayEval) -> EssayEval:
    essay = state["Essay"]

    template = PromptTemplate(
        template= "Evaluate the Language quality of the following essay and provide the feedback and asses a score out of 10 \n Essay : \n{essay}",
        input_variables=["essay"]
    )

    chain = template | structured_model
    result = chain.invoke({"essay":essay})
    return {"language_feedback" :  result.feedback, "individual_score" : [result.score]}

def clarity_feedback(state : EssayEval) -> EssayEval:
    
    essay = state["Essay"]

    template = PromptTemplate(
        template= "Evaluate the clarity of thoght of the following essay and provide the feedback and asses a score out of 10 \n Essay : \n{essay}",
        input_variables=["essay"]
    )

    chain = template | structured_model
    result = chain.invoke({"essay":essay})

    return {"clarity_feedback" : result.feedback, "individual_score" : [result.score]}

def analysis_feedback(state : EssayEval) -> EssayEval:
    essay = state["Essay"]

    template = PromptTemplate(
        template= "Evaluate the depth of analysis of the following essay and provide the feedback and asses a score out of 10 \n Essay : \n{essay}",
        input_variables=["essay"]
    )

    chain = template | structured_model
    result = chain.invoke({"essay":essay})

    return {"analysis_feedback" : result.feedback, "individual_score":[result.score]}

def overall_feedback(state : EssayEval) -> EssayEval:
    
    template = PromptTemplate(
        template = "Based on the provided feedback below, provide the summarized feedback \n Depth Analysis Feedback : \n{analysis} \n\n Clarity of Thought Feedback : \n {clarity} \n\n Quality of Language Feedback : \n {language}",
        input_variables=["analysis", "clarity", "language"]
    )

    chain = template | structured_model

    result = chain.invoke({"analysis" : state["analysis_feedback"], "clarity" : state["clarity_feedback"], "language" : state["language_feedback"]})
    avg_score = sum(state["individual_score"])/len(state["individual_score"])

    return {"overall_feedback" : result.feedback, "avg_score": avg_score} 

graph = StateGraph(EssayEval)

graph.add_node("create_outline", create_outline)
graph.add_node("create_Essay", create_Essay)
graph.add_node("language_feedback", language_feedback)
graph.add_node("clarity_feedback", clarity_feedback)
graph.add_node("analysis_feedback", analysis_feedback)
graph.add_node("overall_feedback", overall_feedback)

graph.add_edge(START,"create_outline")
graph.add_edge("create_outline","create_Essay")
graph.add_edge("create_Essay","language_feedback")
graph.add_edge("create_Essay","clarity_feedback")
graph.add_edge("create_Essay","analysis_feedback")
graph.add_edge("analysis_feedback","overall_feedback")
graph.add_edge("language_feedback","overall_feedback")
graph.add_edge("clarity_feedback","overall_feedback")
graph.add_edge("overall_feedback", END)

workflow = graph.compile()

print(workflow.get_graph().print_ascii())

final_state = workflow.invoke({"topic" : "Rise of AI in INDIA"})

print(final_state["avg_score"])



