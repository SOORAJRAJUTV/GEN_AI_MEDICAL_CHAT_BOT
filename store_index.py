from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore  # ✅ Avoids name conflict
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Load and process data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Ensure index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Ensure this matches your embedding model's output
        metric='cosine',
        spec=ServerlessSpec(  # ✅ Fix: Added required spec argument
            cloud='aws',  
            region='us-east-1'  
        )
    )

# Initialize Pinecone index
index = pc.Index(index_name)

# ✅ Fix: Use correct Pinecone class for LangChain
docsearch = PineconeVectorStore.from_texts(
    [t.page_content for t in text_chunks], 
    embeddings, 
    index_name=index_name
)

print("✅ Indexing completed successfully!")
