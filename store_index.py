from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

qdrant_client = QdrantClient(
    url=os.getenv('url'), 
    api_key=os.getenv('api_key')
)

# print(url)
# print(api_key)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the qdrant
url = os.getenv('url')
api_key = os.getenv('url')


#Creating Embeddings for Each of The Text Chunks & storing

qdrant = Qdrant.from_documents(
    text_chunks,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="medai",
)

