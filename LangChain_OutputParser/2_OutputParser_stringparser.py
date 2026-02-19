from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    #max_new_tokens=250
    )
model = ChatHuggingFace(llm=LLM)

template1 = PromptTemplate(
    template = "Provide detailed history of {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template = "Give me 5 line summary of given text \n {text}",
    input_variables=["text"]
)

Parser = StrOutputParser()

chain = template1 | model | Parser | template2 | model | Parser

result = chain.invoke({"topic" : "India"})

print(result)