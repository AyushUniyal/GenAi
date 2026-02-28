from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """ 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatOpenAI(model="gpt-4")

st.header("Research Summarizer")
user_input = st.text_input("Enter your Prompt")

if st.button("Summarize"):
    result = model.invoke(user_input)
    st.text(result.content)
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

doc = splitter.split_text(text)
print(doc[0])

print("\n ############################## \n")

print(doc[1])

print("\n ############################## \n")

print(doc[2])

