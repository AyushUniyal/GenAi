from langchain_core.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class Muliplication(BaseModel):
    a : int = Field(required=True, description="The first number to multiply")
    b : int = Field(required=True, description="The second number to multiply")

class multiplication(BaseTool):
    name : str = "multiply"
    description : str = "Multiply two numbers"
    args_schema : Type[BaseModel] = Muliplication

    def _run(self, a:int, b:int) -> int:
        return a*b
    
tool = multiplication()

print(tool.invoke({"a":3, "b":5}))