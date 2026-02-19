from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import Field, BaseModel

load_dotenv()

class person(BaseModel):
    name : str = Field(description="name of the person")
    age : int = Field(gt=18, description="Age of the person")
    city : str = Field(description="City of the person")

LLM = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    temperature=1
)

model = ChatHuggingFace(llm=LLM)
parser = PydanticOutputParser(pydantic_object=person)
template = PromptTemplate(
    template = "Give name, Age and city of the fictional peron living in {country} \n{fromat}",
    input_variables=["country"],
    partial_variables={"fromat":parser.get_format_instructions()}
)

prompt = template.invoke({"country" : "India"})

result = model.invoke(prompt)

print(parser.parse(result.content))


#With Chain

chain = template | model | parser
result2 = chain.invoke({"country" : "Australia"})
print(result2)