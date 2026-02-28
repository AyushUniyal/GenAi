from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are a {domain} expert"),
    ("human","Explain me in simple terms, what is {topic}")
])

prompt = chat_template.invoke({
    "domain" : "Cricket",
    "topic" : "Stumps"
})

print(prompt)