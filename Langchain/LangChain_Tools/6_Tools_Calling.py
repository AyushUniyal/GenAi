from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a : int,b : int) -> int:
    """" Multiply two numbers """
    return a*b

llm = ChatOpenAI()
llm_tool = llm.bind_tools([multiply])
query = HumanMessage("Can you multiply 4 with 7 ?")
messages = [query]
result = llm_tool.invoke(messages)
print(result)
messages.append(result)

print(messages)

tool_result = multiply.invoke(result.tool_calls[0])
print("\n\n\n",tool_result)

messages.append(tool_result)

print("\n\n\n",messages)

print("\n\n\n",llm_tool.invoke(messages).content)