from langchain_community.document_loaders import UnstructuredPDFLoader 
# from langchain_community.document_loaders import OnlinePDFLoader
import os
# Load a PDF from a local file

wd = os.getcwd()
print(wd)

pth = 'pdf-rag/res/BOI.pdf'
model = 'llama3.2'

if pth:
    loader = UnstructuredPDFLoader(pth)
    doc = loader.load()
    print("data loaded")
else:
    print("No file path provided")

#see first page
context = doc[0].page_content
# print(cont[:100])

#extract text and split into chunks 

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200)
chunks = text_splitter.split_documents(doc)
print("splitted")

# print(len(chunks),"\n",chunks[0])

# add to vector database
import ollama
# ollama.pull("nomic-embed-text")

db = Chroma.from_documents(
    documents=chunks,
    embedding = OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="test-rag",
)
print("added to db")

# # Query the database
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model=model)

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    db.as_retriever(), llm, prompt=QUERY_PROMPT
)

#Rag prompt

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input="what is the document about?")



print(res)