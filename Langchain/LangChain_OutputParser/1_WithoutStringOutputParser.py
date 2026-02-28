from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation"
)

model = ChatHuggingFace(llm=LLM)

template1 = PromptTemplate(
    template="Write detail history about {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Give me 5 line summary of given text \n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic" : "blackhole"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"text" : result.content})
result2 = model.invoke(prompt2)

print(result2.content)