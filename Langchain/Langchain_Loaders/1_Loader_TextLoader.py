from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model = ChatAnthropic(model_name="claude-haiku-4-5-20251001")

loader = TextLoader('Story.txt', encoding="utf-8")

docs = loader.load()

print(len(docs))

template = PromptTemplate(
    template="What should be the appropriate title for this given text? \n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"text" : docs[0].page_content})
print(result)