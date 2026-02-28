from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

LLM = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation"
)
model = ChatHuggingFace(llm=LLM)

result = model.invoke("What is the capital of India")

print(result)