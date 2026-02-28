from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
import requests as re
import json

load_dotenv()
llm = ChatOpenAI()

searcg_tool = DuckDuckGoSearchResults()

@tool
def weather_tool(city : str) -> dict:
    """" This tool pprovides the weather result of given city """
    API_kry = "e3a8f27a1372acbf0c4191b843d9d8f3"
    url = f"https://api.weatherstack.com/current?access_key={API_kry}&query={city}"
    response = re.get(url)
    return response.json()

print(weather_tool.invoke({"city":"Abohar"}))

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    prompt=prompt,
    tools=[searcg_tool, weather_tool]
)

agent_ation =  AgentExecutor(
    verbose=True,
    agent=agent,
    tools=[searcg_tool, weather_tool]
)

print(agent_ation.invoke({"input":"What is the current population of Jaipur and what's the weather theres?"}))