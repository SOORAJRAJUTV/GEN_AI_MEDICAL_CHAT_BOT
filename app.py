from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
from src.prompt import *
import os

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Set API keys for environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()

# Define Pinecone index name
index_name = "medicalbot"

# Check if Pinecone index exists
try:
    docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
except Exception as e:
    raise ValueError(f"Error loading Pinecone index: {e}")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = GoogleGenerativeAI(model="gemini-pro", temperature=0.4, max_output_tokens=500)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", prompt_template),
        ("human", "Question: {input}\nContext: {context}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("User Input:", input)


    

    # Debug: Check what the retriever is fetching
    retrieved_docs = retriever.invoke(input)
    print("Retrieved Documents:", retrieved_docs)

     # Extract page_content from retrieved_docs to form the context
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    print("Context:", context)

    response = rag_chain.invoke({"input": input, "question": input, "context": context})
    print("Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)