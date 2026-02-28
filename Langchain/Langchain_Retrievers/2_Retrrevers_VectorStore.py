from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document
load_dotenv()

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="Vector_Store",
    collection_name="Sample"
)

print(store.from_documents(docs))

print(store.similarity_search_with_score("What is Langchian?", k=2))
