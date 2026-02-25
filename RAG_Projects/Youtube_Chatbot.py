from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
import sys
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()

template = PromptTemplate(
    template="""
    
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.\n
      Context : {context}\n
      Question: {question}


""",
input_variables=["context","question"]
)

sys.stdout.reconfigure(encoding='utf-8')
ytt_api = YouTubeTranscriptApi()
data = ytt_api.fetch("T-D1OfcDW1M", languages=['en'])
print(type(data))

transcript = "  ".join(chunks.text for chunks in data)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

chunks= splitter.create_documents([transcript])

print(len(chunks))


store = FAISS.from_documents(
    embedding=OpenAIEmbeddings(),
    documents=chunks
)


retriever = store.as_retriever(search_type="similarity",search_kwargs={"k":10})
print("\n\n The answr is \n \n")
# print(retriever.invoke("What is deepmind ?"))

def formatted_context(retrieved_doc):
    return "\n\n".join(doc.page_content for doc in retrieved_doc)


question = "How does RAG helps? "
parser = StrOutputParser()
parallel_chain = RunnableParallel(
    {
        "context" : retriever | RunnableLambda(formatted_context),
        "question" : RunnablePassthrough()
    }
)

# print(parallel_chain.invoke(question))

simple_chain = parallel_chain | template | model | parser

print(simple_chain.invoke(question))