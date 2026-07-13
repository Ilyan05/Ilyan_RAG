from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader

DATA_DIR = "data"

documents = []

documents+= DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls= PyPDFLoader).load()
documents+= DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=Docx2txtLoader).load()
documents+= DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls= TextLoader, loader_kwargs={"encoding":"utf-8"}).load()

print(f"Documents are loaded. Documents :{len(documents)}")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

text_splitter = SemanticChunker(embeddings=EMBEDDING, breakpoint_threshold_type="percentile")

chunks = text_splitter.split_documents(documents)

print("Sementic chunks created")

import os
from pathlib import Path
from langchain_chroma import Chroma

CHROMA_DIR = "chroma_db"

if Path(CHROMA_DIR).exists():
    print("Loading ChromaDB Database")
    vector_db = Chroma(embedding_function=EMBEDDING,persist_directory=CHROMA_DIR)
else:
    print("Creating new ChromaDB Database")
    vector_db = Chroma.from_documents(documents=chunks,persist_directory=CHROMA_DIR, embedding=EMBEDDING)


question = input("Enter your question")
TOP_K = 3

retriever = vector_db.as_retriever(search_kwargs = {"k":TOP_K})
retrieved_docs = retriever.invoke(question)

context = "\n\n".join(doc.page_content for doc in retrieved_docs)

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_TOKEN")
client = genai.Client(api_key=api_key)

while True:
    if question.lower() == ["exit","stop"]:
        print("Stopped")
        break
    
    prompt =f"""
You are a Porsche Enthusiast and an expert in porsche models, and the great history.
Tell me, {question}, from the context of : {context}
""" 
    response = client.models.generate_content(model = "gemini-3.1-flash-lite", contents= prompt)


print("\n Gemini Response: \n")
print(response.text)
        