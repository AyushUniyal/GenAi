from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.langchain.com/langsmith/agent-builder-quickstart")

docs = loader.load()
print(docs[0].metadata)