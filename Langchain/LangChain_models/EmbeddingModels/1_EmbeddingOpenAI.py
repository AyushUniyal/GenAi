from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

Embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)
result = Embedding.embed_query("The Capital of India is New Delhi")

print(result)
print(str(result))