from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

load_dotenv()

def load_pdf(folder_path):
    loader = PyPDFDirectoryLoader(folder_path)
    return loader.load()

def splitter_pdf(pdf):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    docs = splitter.split_documents(pdf)
    # for i, doc in enumerate(docs):
    #     print(f"--- Chunk {i+1} ---")
    #     print(doc.page_content)
    #     print()

    return docs

def create_vectorstore(chunks):
    vectorestore = FAISS.from_documents(
        embedding= OpenAIEmbeddings(model="text-embedding-3-small"),
        documents=chunks
    )
    vectorestore.save_local("FAISS_store")
    return vectorestore

def load_vectorstore(path):
    return FAISS.load_local(path, OpenAIEmbeddings(model="text-embedding-3-small"), allow_dangerous_deserialization=True)

def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs={"k":4}
    )
    
def rewrite_query(question, chathistory):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0) 
    template = PromptTemplate(
        template="Given the conversation history, rewrite the question as a standalone search query \n question: {question} \n conversation history : {chathistory} ",
        input_variables=["question","chathistory"]
    )

    chain = template | model | StrOutputParser()
    return chain.invoke({"question": question, "chathistory":chathistory})   

def rag_chain(chathistory, question, retriever):
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()
    
    # prompt = PromptTemplate(
    #     template="""You are an Helpful assistant which help user to provide help based on the provided context from the resume. answer only from the context, if you don't know say I don't know \n Context: {context} \nQuestion: {question} """,
    #     input_variables=["context","question"]
    # )

    prompt=ChatPromptTemplate([
        ("system", """You are a helpful assistant for resume screening.
You have access to resumes from multiple candidates.
Each piece of context is tagged with its source file.

STRICT RULES:
- Only attribute skills/experience to a candidate if their resume explicitly states it
- If comparing candidates, only compare what is explicitly mentioned in each candidate's context
- If a candidate's resume doesn't mention something, say so explicitly — never assume or infer
- Always mention which resume your answer is based on

Context: {context}"""),
        (MessagesPlaceholder(variable_name='chathistory')),
        ("human", '{question}')
    ])

    retrieved_doc = retriever.invoke(question)
    contexts_with_source = [
    f"[Source: {doc.metadata['source']}]\n{doc.page_content}" 
    for doc in retrieved_doc]
    context_str = "\n\n".join(contexts_with_source)

    
    # parallel = RunnableParallel(
    #     {   
    #         "question" : RunnableLambda(lambda x : x["question"]),
    #         "context" : RunnableLambda(lambda x : x["question"]) | retriever | RunnableLambda(lambda x:"\n\n".join(doc.page_content for doc in x)),
    #         "chathistory" : RunnableLambda(lambda x : x["chathistory"])
    #     }
    # )

    chain = prompt | model | parser

    answer = chain.invoke({"question":question, "chathistory":chathistory, "context" : context_str})
    contexts_list = [doc.page_content for doc in retrieved_doc]
    return {"answer":answer, "contexts":contexts_list}

def evaluate_rag(questions, answers, contexts):
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }
    dataset = Dataset.from_dict(data)
    return evaluate(dataset, metrics=[faithfulness, answer_relevancy])

folder = "resume/"
question  = "What is the name and qualification of the candidate?"     
loaded_pdf = load_pdf(folder)

questions=[]
answers=[]
contexts=[]

if os.path.exists("FAISS_store"):
    laoded_store = load_vectorstore("FAISS_store")
else:
    splitted_pdf = splitter_pdf(loaded_pdf)
    laoded_store = create_vectorstore(splitted_pdf)

retriever = get_retriever(laoded_store)
chat_history = []

retrieved_doc = retriever.invoke("elastic stack experience")
# for doc in retrieved_doc:
#     print(doc.metadata)
#     print(doc.page_content[:100])
#     print()
while True:
    question = input("You : ")
    if question.lower() in ["quit", "exit"]:
        break
    else:
        chat_history.append(HumanMessage(content=question))
        rewritten = rewrite_query(question,chat_history) if chat_history else question
        result = rag_chain(chat_history, rewritten, retriever)
        answer = result["answer"]
        retrieved_contexts = result["contexts"]
        
        questions.append(question)
        answers.append(answer)
        contexts.append(retrieved_contexts)
        
        chat_history.append(AIMessage(content=answer))
        print(f"Assistant : {answer}\n")

scores = evaluate_rag(questions, answers, contexts)
print(scores)