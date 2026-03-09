from pprint import pp
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model = 'gpt-4o', temperature=1.5, max_completion_tokens=100)
result = model.invoke("Who won the cricket cup world finals 2026?")

pp(result.content)