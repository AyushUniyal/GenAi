from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)

model = ChatHuggingFace(llm=LLM)

template = PromptTemplate(
    template="Some 5 interesting facts on {topic}",
    input_variables=["topic"]
)
parser = StrOutputParser()
chain = template | model | parser

result = chain.invoke({"topic" : "LLM Model"})
print(result)

chain.get_graph().print_ascii()
