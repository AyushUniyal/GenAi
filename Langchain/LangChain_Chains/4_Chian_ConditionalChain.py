from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
import os 

os.environ["LANGCHAIN_PROJECT"] = "Conditional Chain"

config = {
    "run_name" : "Condiotnal Chain",
    "tags" : ["Conditional Chaining", "Langchain", "Practice"],
    "metadata" : {
        "model" : "ChatOpen AI",
        "project" : "GenAI/Langchain/Langchain_Chains/4_Chain_ConditionalChain.py",
        "purpose" : "Sentiment Analyzer"
    }
}

load_dotenv()

model = ChatOpenAI()

class feedback(BaseModel):
    sentiment : Literal["Positive", "Negative"] = Field(description="User sentiment in positive or negative")
    key_highlights : list[str] = Field(description="Key Highlights provided by the user")
    # review : str = Field(description="based on the user sentiment and key highlight generate a review")

parser = PydanticOutputParser(pydantic_object=feedback)
Jparser = JsonOutputParser()

template1 = PromptTemplate(
    template="classify the sentiment(positive or negative) of user from the given feedback \n {feedback} \n {fromat}",
    input_variables=["feedback"],
    partial_variables={"fromat" : parser.get_format_instructions()}
)

classifier_chain = template1 | model | parser

template2 = PromptTemplate(
    template="Write an appropriate reponse to this positive feedback and user mentioned below highlights \n {key_highlights}",
    input_variables=["key_highlights"]
)

template3 = PromptTemplate(
    template="Write an appropriate reponse to this negative feedback  and user mentioned below highlights \n {key_highlights}",
    input_variables=["key_highlights"]
)

parser1 = StrOutputParser()

reviewparser = RunnableLambda(lambda x:x.model_dump())

branch = RunnableBranch(
    (lambda x:x["sentiment"] == "Positive" , template2 | model | parser1),
    (lambda x:x["sentiment"] == "Negative", template3 | model | parser1),
    RunnableLambda(lambda x: "Sentiment is incorrect")
)

chain = classifier_chain | reviewparser | branch

result = chain.invoke({"feedback" : "This is very laggy mobile phone, the Display is very dull and UI is unresponsive"}, config=config)
print(result)