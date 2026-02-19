from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation"
)
model = ChatHuggingFace(llm=LLM)
parser = JsonOutputParser()

template = PromptTemplate(
    template="Give the five fictional character \n {formatting}",
    input_variables=[],
    partial_variables={"formatting" : parser.get_format_instructions()}
)

#with using chain

chain = template | model | parser

result = chain.invoke({})

print(result)

#Without using chain 

# prompt = template.invoke({})
# result = model.invoke(prompt)
# print(parser.parse(result.content))
