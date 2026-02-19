from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("artificial_intelligence_tutorial.pdf")
docs = loader.load()

print(docs[1].metadata)
print(len(docs))