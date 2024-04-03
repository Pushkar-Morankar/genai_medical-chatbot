from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

qdrant_client = QdrantClient(
    url=url, 
    api_key=api_key
)

# print(url)
# print(api_key)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the qdrant
url = "https://ca57a25b-6ae0-4dd7-b454-839c21d31eb5.us-east4-0.gcp.cloud.qdrant.io:6333"
api_key = "NCs4WeHRA9Nb9CZUh35nmRQ4--KwoqoP4ApP2zC9BFQqmAM2-YoKPw"


#Creating Embeddings for Each of The Text Chunks & storing

qdrant = Qdrant.from_documents(
    text_chunks,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="medai",
)

