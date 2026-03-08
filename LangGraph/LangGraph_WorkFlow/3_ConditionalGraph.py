from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class sentiment_struc(BaseModel):
    sentiment : Literal["positive", "negative"] = Field(description="Sentiment of the feedback, must be positive or negative")

class diagnose_struct(BaseModel):
    issue_type : Literal["UX", "performance", "BUG", "Support", "Other"] = Field(description="The Category of issue mentioned in the review")
    tone : Literal["angry", "frustated", "dissapointed", "calm"] = Field(description="Emotional tone expressed by the user")
    urgency : Literal["low", "medium", "high"] = Field(description="how urgent or critical the issue is.")

model = ChatOpenAI()

sentiment_model = model.with_structured_output(sentiment_struc)
diagnose_model = model.with_structured_output(diagnose_struct)

class feedbacState(TypedDict):
    feedback : str
    sentiment : Literal["positive", "negative"]

    diagnosis : dict
    response : str

def sentiment(state : feedbacState) -> feedbacState:

    template = PromptTemplate(
        template="Based on the user's feedback evaluate the sentiment of the user \n Feedback : \n {feedback}",
        input_variables=["feedback"]
    )

    chain = template | sentiment_model
    result = chain.invoke({"feedback" : state["feedback"]})
    return {"sentiment" : result.sentiment}

def diagnosis(state : feedbacState) -> feedbacState:
    
    template = PromptTemplate(
        template="Diagnose the issue type based on the negative review \n Review \n {feedback} \n State tone, urgency and issue",
        input_variables=["feedback"]
    )

    chain = template | diagnose_model
    result = chain.invoke({"feedback":state["feedback"]})

    return {"diagnosis": result.model_dump()}

def positive_response(state: feedbacState) -> feedbacState:

    template = PromptTemplate(
        template= "Write a warm thank you message in reponse to below feedback \n Feedback \n {feedback}.\n\n And also kindly ask user to leave feedback on the website",
        input_variables=["feedback"]
    )

    chain = template | model
    result = chain.invoke({"feedback" : state["feedback"]})

    return {"response" : result.content}

def negative_response(state : feedbacState) -> feedbacState:

    template =  PromptTemplate(
        template = "You are a support assiatant, user had a {issue} issue, sounded {tone} and marked urgency as {urgency}\n Write an empathatic and helpful resolution message.",
        input_variables=["issue", "tone", "urgency"]
    )

    diagnosis = state["diagnosis"]

    chain = template | model
    result = chain.invoke({"issue" : diagnosis["issue_type"], "tone" : diagnosis["tone"], "urgency" : diagnosis["urgency"]})
    return {"response":result.content}

def check_condition(state : feedbacState) -> Literal["positive_response","diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else :
        return "diagnosis"

graph = StateGraph(feedbacState)

graph.add_node("check_sentiment", sentiment)
graph.add_node("diagnosis", diagnosis)
graph.add_node("positive_response", positive_response)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "check_sentiment")
graph.add_conditional_edges("check_sentiment", check_condition)
graph.add_edge("positive_response", END)
graph.add_edge("diagnosis","negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

feedback = """This is hands down the worst smartphone experience I’ve ever had.

From day one, this phone has been nothing but a headache. The battery drains ridiculously fast — I charge it to 100% and within a few hours of normal use, it’s already below 40%. What’s the point of advertising “long-lasting battery” when it can’t even survive a single workday?

The phone constantly lags. Apps freeze, the screen becomes unresponsive, and sometimes it randomly restarts on its own. I’ve missed important calls because the device just decided to hang. This is completely unacceptable for a phone in this price range.

The camera quality is also overhyped. The pictures look grainy in normal indoor lighting, and the so-called “AI enhancement” makes photos look artificial and oversharpened. I didn’t pay this much for mediocre results.

On top of that, the device heats up even while doing basic tasks like browsing or watching videos. It becomes uncomfortable to hold. Is this supposed to be normal?

I am extremely disappointed and frustrated. This product is clearly not performing as promised. I need this issue addressed immediately. Either provide a proper fix (software update or replacement) or arrange a refund as soon as possible. This situation needs urgent resolution.
"""

print(workflow.get_graph().print_ascii())

final_state = workflow.invoke({"feedback": feedback})

print(final_state)