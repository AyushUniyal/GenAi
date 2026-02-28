from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ("system", "You are an expert customer support executive"),
    (MessagesPlaceholder(variable_name='chat_history')),
    ("human", '{query}')
])

chat_history=[]

with open("chat_history.txt") as f:
    chat_history.extend(f.readlines())

prompt = chat_template.invoke({
    "chat_history" : chat_history,
    "query" :  "Didn't received my refund yet"
})

print(prompt)