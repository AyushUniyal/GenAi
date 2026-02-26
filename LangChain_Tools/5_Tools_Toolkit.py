from langchain_core.tools import tool

@tool
def multiply(a:int, b:int) -> int:
    """" Multiply two numbers """
    return a*b

@tool
def add(a:int, b:int) -> int:
    """" Addition of two numbers """
    return a+b

class MathToolkit:
    def get_tools(self):
        return(add, multiply)

toolkit = MathToolkit()
tools = toolkit.get_tools()

for t in tools:
    print(t.name,"=>",t.description)