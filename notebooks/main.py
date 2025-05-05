# 1. Libraries
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath('../src'))

# Modules
from modules.ingesting import ingesting_PDF
from modules.vector_store_module    import chunk_documents, add_to_vector_store

# Retrival (summirising):
from langchain.prompts                  import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers      import StrOutputParser
from langchain_core.runnables           import RunnablePassthrough
from langchain_ollama                   import ChatOllama
from langchain.retrievers.multi_query   import MultiQueryRetriever

# 2. Loading docs
print('Loading the document...')
documents = ingesting_PDF('../data/PDFs/Sustainable_development_of_distance_learning_in_continuing_adult_education__The impact_of_artificial_intelligence.pdf')

# Embedding process
# Chunking
# Adding to the database (Chroma)
# RAG (retrival - Summirisation)

# 3. Chunk the documents
chunks = chunk_documents(documents)

# 4. Add chunks to the vector store
vector_store = add_to_vector_store(chunks)

# 5. RAG (retrival - Summirisation)

# 6. LLM from Ollama
llm = ChatOllama(model = 'gemma2') # gemma2 is faster, but llama3.2 is more comprehe

# 7. Prompt
prompt = PromptTemplate(
    input_variables=['question'],
    template = """You are an AI assistant. Your task is to summarise the text coming from the input PDF, providing a
    summary in the form of a paragraph. The summary should be a concise and coherent representation of the main ideas of the text, including the 
    analysis of the text where the main points and the key arguments are shown. Additionally, the user will prompt a question and the second 
    task you will do after providing a summary is to generate five different versions of the given questions to retrieve the relevant documents 
    from a vector database. By generating multiple perspectives on the user's question, you can help the user to explore the text from 
    different angles, aiming to help the final goal of overcoming the limitations of the distance-based similarity search. Provide these 
    alternative questions separated by a new line. Original question: {question}""",
)

# 8. Retrival
retriever = MultiQueryRetriever.from_llm(
    vector_store.as_retriever(),
    llm,
    prompt = prompt,
)
     
# 9. RAG prompt (to answer only on the given context):
template = """Summarise and Answer the questions ONLY on the following context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 10. Chain (Joining all the answers)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
   
# 11. Running the chain (Chatting): 

# In this section, to enable interactivity, the following command can be used:
# response = chain.invoke({"question": input("Enter the question:\n")})
# However, for the sake of simplicity, the question is provided directly in the notebook.

# Question 1
response = chain.invoke('What is the central theme of the paper?')
print(response)

# Question 2
response = chain.invoke('How is AI changing adult education?')
print(response)

# Question 3
response = chain.invoke('What is "adaptive learning" in AI contexts?')
print(response)

# Question 4
response = chain.invoke('How can institutions ensure ethical AI use?')
print(response)

# Question 5
response = chain.invoke('How is AI reshaping teacher roles?')
print(response)

# Question 6
response = chain.invoke('What is the role of educators in AI-integrated systems?')
print(response)