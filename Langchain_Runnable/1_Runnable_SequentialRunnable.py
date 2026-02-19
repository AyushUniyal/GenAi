from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=LLM)

template1 = PromptTemplate(
    template=" Write a joke on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "Explain the joke, In first give the joke itself and in second line start explaining \n joke is {joke}",
    input_variables=["joke"]
)

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, template2, model, parser)
result = chain.invoke({"topic" : "AI"})
print(result)