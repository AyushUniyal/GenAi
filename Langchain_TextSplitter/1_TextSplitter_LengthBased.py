from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../Books/ITI_AgenticAI_Final.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap=10,
    separator=""
)

doc = splitter.split_documents(docs)

print(doc)
