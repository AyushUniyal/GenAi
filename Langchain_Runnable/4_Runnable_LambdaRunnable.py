from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=LLM)

template1 = PromptTemplate(
    template="Write a joke on {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

def count(input_data):
    return len(input_data.split())

chain = RunnableSequence(template1, model, parser, RunnableParallel(
    {
        "joke" : RunnablePassthrough(),
        "Count" : RunnableLambda(count)
    }
))

print(chain.invoke({"topic" : "reptiles"}))