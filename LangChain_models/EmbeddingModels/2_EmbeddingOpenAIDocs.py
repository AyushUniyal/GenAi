from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=32)

docs = ["India is a great country",
        "India is cultural driven nation",
        "New Delhi is the capital of India"]

result = embeddings.embed_documents(docs)

print(str(result))