from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class Muliplication(BaseModel):
    a : int = Field(required=True, description="The first number to multiply")
    b : int = Field(required=True, description="The second number to multiply")

def multiply(a: int,b: int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name = "multiplication",
    description="Multiply two numbers",
    args_schema=Muliplication

)


print(multiply_tool.invoke({"a":3, "b":5}))