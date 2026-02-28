from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
embedding = OpenAIEmbeddings(model = "text-embedding-3-large", dimensions=300)

documents = [
    "India is a great country with diverse culture.",
    "Delhi is the capital city of India.",
    "Python is widely used for machine learning.",
    "Cricket is the most popular sport in India.",
    "Artificial Intelligence is transforming the world."
]
query = "what is python used for ?"
vector = embedding.embed_documents(documents)

q_vector = embedding.embed_query(query)

result = cosine_similarity([q_vector], vector)[0]
index, score = sorted(list(enumerate(result)), key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("Simillarity score is : ", score)
