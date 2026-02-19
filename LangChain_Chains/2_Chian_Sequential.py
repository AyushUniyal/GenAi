from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm=LLM)

template1 = PromptTemplate(
    template="Write a detailed report on the {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template="Write 5 line pointer on below given text \n {text}",
    input_variables=["text"]
)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic" : "Unemployment in India"})

print(result)