from langchain_community.tools import DuckDuckGoSearchResults

serach_tool = DuckDuckGoSearchResults()
print(serach_tool.invoke("Latest News on Narender Modi"))