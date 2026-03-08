from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
import operator
from pydantic import BaseModel, Field

load_dotenv()



class tweetState(TypedDict):
    topic : str
    tweet : str
    iteration : int
    max_iteration : int
    evaluation : Literal["approved", "need_improvement"]
    feedback: str

    tweet_history : Annotated[list[str], operator.add]
    feedback_hostory : Annotated[list[str], operator.add]

class structured_evaluator(BaseModel):
    feedback : str = Field(...,description="feedback for the tweet")
    evaluation : Literal["approved", "need_improvement"] = Field(...,description="Final evaluation result")

generator = ChatOpenAI(model="gpt-4o-mini")
evaluator = ChatOpenAI(model="gpt-4o")
struc_evaluator = evaluator.with_structured_output(structured_evaluator)
optimizer = ChatOpenAI(model="gpt-4o")

def generate(state : tweetState) -> tweetState:

    messages = [
        SystemMessage(content = "You are a funny and clever Twitter/X influencer."),
        HumanMessage(content = f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Do NOT use question-answer format.
- Max 280 characters.
- Use observational humor, irony, sarcasm, or cultural references.
- Think in meme logic, punchlines, or relatable takes.
- Use simple, day to day english
""")
    ] 

    result = generator.invoke(messages).content

    return {"tweet" : result , "tweet_history":[result]}

def evaluate(state : tweetState) -> tweetState:

    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
Evaluate the following tweet:

Tweet: "{state['tweet']}"

Use the criteria below to evaluate the tweet:

1. Originality – Is this fresh, or have you seen it a hundred times before?  
2. Humor – Did it genuinely make you smile, laugh, or chuckle?  
3. Punchiness – Is it short, sharp, and scroll-stopping?  
4. Virality Potential – Would people retweet or share it?  
5. Format – Is it a well-formed tweet (not a setup-punchline joke, not a Q&A joke, and under 280 characters)?

Auto-reject if:
- It's written in question-answer format (e.g., "Why did..." or "What happens when...")
- It exceeds 280 characters
- It reads like a traditional setup-punchline joke
- Dont end with generic, throwaway, or deflating lines that weaken the humor (e.g., “Masterpieces of the auntie-uncle universe” or vague summaries)

### Respond ONLY in structured format:
- evaluation: "approved" or "need_improvement"  
- feedback: One paragraph explaining the strengths and weaknesses 
""")
    ]

    result = struc_evaluator.invoke(messages)

    return {"feedback" : result.feedback, "evaluation" : result.evaluation, "feedback_hostory":[result.feedback]}

def optimize(state : tweetState) -> tweetState:

    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
Improve the tweet based on this feedback:
"{state['feedback']}"

Topic: "{state['topic']}"
Original Tweet:
{state['tweet']}

Re-write it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
""")
    ]

    result = optimizer.invoke(messages).content
    iteration = state["iteration"] + 1
    return {"tweet": result, "tweet_history":[result], "iteration" : iteration}

def condition_check(state : tweetState) -> Literal["approved", "need_improvement"]:
    if state["evaluation"] == "need_improvement" or state["iteration"] >= state["max_iteration"]:
        return "need_improvement"
    else:
        return "approved"

graph = StateGraph(tweetState)

graph.add_node("generator", generate)
graph.add_node("evaluator", evaluate)
graph.add_node("optimizing", optimize)

graph.add_edge(START, "generator")
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges("evaluator", condition_check, {"approved": END, "need_improvement" : "optimizing"})
graph.add_edge("optimizing","evaluator")

workflow = graph.compile()
print(workflow.get_graph().print_ascii())

initial_state = {
    "topic": "Random Things",
    "iteration": 1,
    "max_iteration": 5
}

final_state = workflow.invoke(initial_state)
print(final_state)

for tweet in final_state["tweet_history"]:
    print("\n",tweet)
    print("\n")