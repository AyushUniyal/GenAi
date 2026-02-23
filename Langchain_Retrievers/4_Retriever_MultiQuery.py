from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

all_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

query = "How to improve energy levels and maintain balance?"

store = FAISS.from_documents(
    embedding=OpenAIEmbeddings(),
    documents=all_docs
)

similarity = store.as_retriever(search_type="similarity", search_kwargs={"k":5})

multi = MultiQueryRetriever.from_llm(
    llm=ChatOpenAI(),
    retriever=store.as_retriever(search_type="mmr", search_kwargs={"k":5,"lambda_mult":0.7})
)

simple = similarity.invoke(query)
multi_ret = multi.invoke(query)

for i, doc in enumerate(simple):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)

for i, doc in enumerate(multi_ret):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
