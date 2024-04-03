from flask import Flask, render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import qdrant
import qdrant_client
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import prompt_template
import os

app= Flask(__name__)


load_dotenv()

url=url, 
api_key=api_key


embeddings = download_hugging_face_embeddings()

#Initializing the Qdrant

qdrant_client = QdrantClient(
    url=url, 
    api_key=api_key
)

#Loading the index

url = url
api_key = api_key


found_docs = qdrant.similarity_search(query)

llm=CTransformers(model="TheBloke/Llama-2-13B-chat-GGML",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=qdrant.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


