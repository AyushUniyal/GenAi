import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from ResumeReader import load_vectorstore, get_retriever, rewrite_query, rag_chain, evaluate_rag
import numpy as np

load_dotenv()

@st.cache_resource
def load_retriever():
    store = load_vectorstore("FAISS_store")
    return get_retriever(store)

retriever = load_retriever()
st.title("Resume Screener 🎯")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers" not in st.session_state:
    st.session_state.answers = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []

for message in st.session_state.chat_history:
    role = "human" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(message.content)

question = st.chat_input("Ask about the candidates...")
if question:
    with st.chat_message("human"):
        st.write(question)

    rewritten = rewrite_query(question, st.session_state.chat_history) if st.session_state.chat_history else question
    result = rag_chain(st.session_state.chat_history, rewritten, retriever)
    answer = result["answer"]
    st.session_state.contexts.append(result["contexts"])

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.session_state.questions.append(question)
    st.session_state.answers.append(answer)

with st.sidebar:
    st.header("RAG Evaluation 📊")
    st.write(f"Conversations collected: {len(st.session_state.questions)}")
    
    if st.button("Evaluate RAG"):
        if len(st.session_state.questions) == 0:
            st.warning("Have at least one conversation first!")
        else:
            with st.spinner("Evaluating..."):
                scores = evaluate_rag(
                    st.session_state.questions,
                    st.session_state.answers,
                    st.session_state.contexts
                )
            st.metric("Faithfulness", f"{np.mean(scores['faithfulness']):.2f}")
            st.metric("Answer Relevancy", f"{np.mean(scores['answer_relevancy']):.2f}")