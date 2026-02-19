from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

LLm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model1 = ChatHuggingFace(llm = LLm)

model = ChatAnthropic(model_name="claude-haiku-4-5-20251001")

template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="Summarise the report in 10 lines \n {report}",
    input_variables=["report"]
)

parser = StrOutputParser()

chain = RunnableSequence(template1, model, parser, RunnableBranch(
    (lambda x:len(x.split())>100, RunnableSequence(template2, model, parser)),
    RunnablePassthrough()
))

print(chain.invoke({"topic" : "Russia vs America"}))