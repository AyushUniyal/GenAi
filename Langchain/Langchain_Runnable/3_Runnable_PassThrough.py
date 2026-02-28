from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=LLM)

template1 = PromptTemplate(
    template = "Write a joke on {topic}",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template = "Explain the {joke}",
    input_variables=["joke"]
)


parser = StrOutputParser()

sequence = RunnableSequence(template1, model, parser)

chain = RunnableSequence(sequence, RunnableParallel(
    {
        "joke" : RunnablePassthrough(),
        "explanation" : RunnableSequence(template2, model, parser)
    }
))

print(chain.invoke({"topic" : "AI"}))