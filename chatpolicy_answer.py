from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import torch
from sentence_transformers import SentenceTransformer, util
import os 
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import  FAISS
import fitz
import openai
from langchain.document_loaders import TextLoader
#os.environ["OPENAI_API_KEY"] = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXX"

from dotenv import load_dotenv, dotenv_values
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

loader = TextLoader("raw_text.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=0,
                                      length_function = len )
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(search_kwargs={"k": 3}), memory=memory)


def get_answer(query):
    result = qa({"question": query})
    return result["answer"]

#############################



# Load a pre-trained Sentence Transformer model
model_name = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
pdf_file_path='g4-4f.pdf'

def build_vectorDB(pdf_file_path='g4-4f.pdf' ):
    
    reader = PdfReader(pdf_file_path)

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 20,
        chunk_overlap  = 2,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    return docsearch


# Function to perform semantic search
def semantic_search(query, database_embeddings, sentences, top_k=1):

    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarities between the query and database embeddings
    cos_scores = util.pytorch_cos_sim(query_embedding, database_embeddings)[0]

    # Get the indices of the top-k most similar documents
    top_results = torch.topk(cos_scores, k=top_k)

    # Print the most similar documents and their scores
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"Similarity: {score:.4f}")
        print(f"Document: {sentences[idx]}")
        print("=" * 50)
    return top_results


def parse_pdf(file_name, query):
    doc = fitz.open(file_name)

    co_ords = {}
    for idx, page in enumerate(doc):
        text_instances = page.search_for(query)
        if text_instances:
            x0_coords = [rect.x0 for rect in text_instances]
            y0_coords = [rect.y0 for rect in text_instances]
            x1_coords = [rect.x1 for rect in text_instances]
            y1_coords = [rect.y1 for rect in text_instances]

    # Extract coordinates (left, top, right, bottom) of each text instance
            larger_rect = fitz.Rect(min(x0_coords), min(y0_coords), max(x1_coords), max(y1_coords))

            # Create a highlight annotation with a yellow color (1, 1, 0)
            highlight = page.add_highlight_annot(larger_rect)
            doc.save("output.pdf")
            

            co_ords[idx] = larger_rect

    doc.close()
    print(co_ords)

    return co_ords

db = build_vectorDB(pdf_file_path)


