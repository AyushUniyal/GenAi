from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task = "text-generation"
)

model = ChatHuggingFace(llm = LLM)

template1 = PromptTemplate(
    template = "Generate a tweet about {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template = "Generate a linked in post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "X tweet" : RunnableSequence(template1, model, parser),
        "LinkedIn Post" : RunnableSequence(template2, model, parser) 
    }
)

result = chain.invoke({"topic" : "langchian"})
print(result["X tweet"])
print("\n\n")
print(result["LinkedIn Post"])