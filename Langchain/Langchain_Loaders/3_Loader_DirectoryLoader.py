from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="Books",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for d in docs:
    print(d.page_content)