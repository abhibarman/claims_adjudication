import fitz
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import pandas as pd
import chromadb

# Load the CSV data into a DataFrame
awac_data = pd.read_csv("awac_data.csv")


api_key = "sk-"
pdf_file_path = "g4-4f.pdf"
chain = None

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def sanitize_collection_name(file_name: str) -> str:
    # Remove file extension
    name_without_extension = file_name.split('.')[0]
    
    # Replace non-alphanumeric characters with hyphen, but not consecutive hyphens
    sanitized_name = re.sub('[^a-zA-Z0-9]+', '-', name_without_extension)
    
    # Ensure it starts with an alphanumeric character
    if not sanitized_name[0].isalnum():
        sanitized_name = "a" + sanitized_name[1:]
    
    # Ensure it ends with an alphanumeric character
    if not sanitized_name[-1].isalnum():
        sanitized_name = sanitized_name[:-1] + "a"
    sanitized_name = sanitized_name.replace("-", "_")

    # Ensure the length is between 3 and 63 characters
    return sanitized_name[:63]

def build_chain() -> ConversationalRetrievalChain:
    # Load PDF content and sanitize the file name for the collection
    pdf_doc = fitz.open(pdf_file_path)
    pdf_content = {page_num: page.get_text() for page_num, page in enumerate(pdf_doc)}
    collection_name = sanitize_collection_name(pdf_file_path)

    # Initialize a Chroma client
    client = chromadb.Client()

    # Create or get an existing collection
    collection = client.get_or_create_collection(name=collection_name)

    # Load embeddings model
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Convert the parsed content to a list of Document objects
    pdf_documents = [Document(content, {"page": page_num}) 
                for page_num, content in pdf_content.items()]
    
    awac_documents = [Document(row["page_content"], {"page":index}) for index, row in awac_data.iterrows()]

    documents = pdf_documents + awac_documents

    # Create the Chroma index from the PDF content
    pdf_search = Chroma.from_documents(
        documents=documents,
        embeddings=embeddings,
        collection_name=collection_name,
    )

    # Initialize the ChatOpenAI model
    chat_model = ChatOpenAI(temperature=0.0, openai_api_key=api_key)

    # Create the ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        chat_model,
        retriever=pdf_search.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
    )

    return chain

def retrieve_search_results_withAWAC(user_query: str) -> pd.DataFrame:

    global chain

    if chain is None:
        chain = build_chain()
    # Get the search results for the user query
    results = chain({"System":"Please answer the following user question based on the available information. If you are not sure of the answer please mention that you could not find the answer. ","question": user_query,"chat_history":""})    
    return results['answer']


""" 
#Example usage:
api_key = os.getenv("OPENAI_API_KEY")
user_query = "Give me details of the vehicle"
pdf_file_path = "g4-4f.pdf"
results = retrieve_search_results(user_query)
print(results)

user_query = "Any emergency service included ?"
results = retrieve_search_results(user_query)
print(results)

user_query = "Tell me about Allied Insurance ?"
results = retrieve_search_results(user_query)
print(results)

user_query = "Give the contact details?"
results = retrieve_search_results(user_query)
print(results) """