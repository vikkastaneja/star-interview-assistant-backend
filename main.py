from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os
import time

PDF_INDEX_PATH = "data"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral"

app = FastAPI()

# Enable CORS (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector DB once at startup
embedding = OllamaEmbeddings(model=EMBED_MODEL)
vector_db = FAISS.load_local(
    folder_path=PDF_INDEX_PATH,
    embeddings=embedding,
    allow_dangerous_deserialization=True  # ✅ you trust your own files
)

llm = Ollama(model=LLM_MODEL)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_db.as_retriever())

print("🔥 Warming up Ollama...")
_ = qa.invoke("Hello")

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_star_response(query: Query):
    print("⏱️ Received question:", query.question)
    start = time.time()
    prompt = f"""Answer the following question using the STAR method with numbers and statistics where possible.:
    
**S** - Describe the *situation* or background.

**T** - Define the *task* or objective.

**A** - Explain the *action* you took.

**R** - State the *result* or outcome.

Question: {query.question}

Respond in the STAR format only.
"""

    result = qa.invoke(prompt)
    print(f"✅ Responded in {time.time() - start:.2f} seconds")
    return {"result": result}
