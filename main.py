
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama

import os
import time

PDF_INDEX_PATH = "data"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:instruct"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector DB at startup
embedding = OllamaEmbeddings(model=EMBED_MODEL)
vector_db = FAISS.load_local(
    folder_path=PDF_INDEX_PATH,
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

llm = Ollama(model=LLM_MODEL)
retriever = vector_db.as_retriever()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_star_response(query: Query):
    print("⏱️ Received question:", query.question)
    start = time.time()

    docs = retriever.get_relevant_documents(query.question)
    context = "\n\n".join(doc.page_content for doc in docs[:3])

    prompt = f"""Answer the following interview question in STAR format using the given CONTEXT. Do not make up facts that are not in the context.

Question: {query.question}

Context:
{context}

Respond strictly in this format:
**Situation**: [A specific situation, including team size, deadline, or tools involved]
**Task**: [Your responsibility or challenge]
**Action**: [Steps you took — include leadership, technical, or collaboration aspects]
**Result**: [Measurable outcomes with numbers if possible, recognition, or impact]

Only respond with the STAR story. Use "I" instead of "the user". Do not explain STAR or include commentary."""

    full_response = ""
    for chunk in llm.stream(prompt):
        full_response += chunk

    print(f"✅ Responded in {time.time() - start:.2f} seconds")
    return {"result": full_response}


@app.post("/ask-stream")
async def ask_stream(query: Query):
    print("🎤 Streaming request received:", query.question)

    docs = retriever.get_relevant_documents(query.question)
    context = "\n\n".join(doc.page_content for doc in docs[:3])

    prompt = f"""Answer the following interview question in STAR format using the given CONTEXT. Do not make up facts that are not in the context.

Question: {query.question}

Context:
{context}

Respond strictly in this format:
**Situation**: [A specific situation, including team size, deadline, or tools involved]
**Task**: [Your responsibility or challenge]
**Action**: [Steps you took — include leadership, technical, or collaboration aspects]
**Result**: [Measurable outcomes with numbers if possible, recognition, or impact]

Only respond with the STAR story. Use "I" instead of "the user". Do not explain STAR or include commentary."""

    def generate():
        for chunk in llm.stream(prompt):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
