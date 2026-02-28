from langchain_core.tools import tool, InjectedToolArg
from langchain_openai import ChatOpenAI
from typing import Annotated
from langchain_core.messages import HumanMessage
import requests as re
import json
from dotenv import load_dotenv

load_dotenv()

@tool
def get_currency_rate(base_curr: str, target_curr: str) -> float:
    """" Function takes Base Currency and target currency and returns current exchange rate """
    api_key = "f4ac4af139dc297bff5df71a"
    url= f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_curr}/{target_curr}"
    response = re.get(url)
    return response.json()

@tool
def convert(amount : int, exchange_rate : Annotated[float, InjectedToolArg]) -> float:
    """" this function takes the amount for which conversion is required and multiple with exchange rate """
    return amount*exchange_rate


llm = ChatOpenAI()

llm_tool = llm.bind_tools([get_currency_rate, convert])

query = HumanMessage("What is the exchange rate of INR w.r.t USD and how much 54 INR will be in USD ?")
messages = [query]

result = llm_tool.invoke(messages)
messages.append(result)


tool_calls = result.tool_calls

for tool in tool_calls:
    if tool["name"] == "get_currency_rate":
        tool_msg1 = get_currency_rate.invoke(tool)
        exchange_rate = json.loads(tool_msg1.content)["conversion_rate"]
        messages.append(tool_msg1)

    if tool["name"] == "convert":
        tool["args"]["exchange_rate"] = exchange_rate
        tool_msg2 = convert.invoke(tool)
        messages.append(tool_msg2)

print(llm_tool.invoke(messages).content)
       
