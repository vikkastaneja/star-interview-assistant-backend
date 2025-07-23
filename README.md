# 🎤 STAR Interview Assistant – Backend

This is the backend API for the **STAR Interview Assistant**. It allows you to ask behavioral interview questions and receive answers in the STAR format (Situation, Task, Action, Result), powered by your own PDF knowledge base using Ollama + LangChain.

---

## 🚀 Features

- Accepts natural language questions via REST API
- Uses LangChain + Ollama (e.g. Mistral) as the LLM backend
- Retrieves relevant context from your PDF files via FAISS
- Automatically responds using STAR method
- Designed for local or embedded frontend clients (e.g., React, Electron)

---

## 🧱 Technology Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [LangChain](https://www.langchain.com/)
- [Ollama](https://ollama.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- PDF parsing via `PyPDF2`

---

## 🗂 Folder Structure

```
star-backend/
├── main.py              # FastAPI app with STAR response logic
├── data/                # FAISS vector store (auto-created)
├── pdfs/                # Place your PDF files here
└── requirements.txt     # Python dependencies
```

---

## 📦 Setup Instructions

### 1. Clone the repo and create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
fastapi
uvicorn
langchain
langchain-ollama
faiss-cpu
PyPDF2
```

---

### 3. Install and run [Ollama](https://ollama.com/)

Install Ollama from https://ollama.com

Then pull a model (e.g. Mistral):

```bash
ollama pull mistral
```

Keep it running in the background.

---

### 4. Add your PDFs

Put your resume, case studies, or other behavioral content inside the `pdfs/` directory. On first run, a FAISS vector store will be generated from your data.

---

### 5. Run the backend

```bash
uvicorn main:app --reload
```

Backend will be available at: [http://localhost:8000](http://localhost:8000)

---

## 🔗 API Reference

### `POST /ask`

Submit a question and get a STAR-format answer.

#### Request:
```json
{
  "question": "Tell me about a time you led under pressure"
}
```

#### Response:
```json
{
  "answer": "S: ...\nT: ...\nA: ...\nR: ..."
}
```

---

## 🧠 Notes & Customization

- **Model:** You can switch from `mistral` to `llama2`, `gemma`, or any Ollama-supported model in `main.py`.
- **Prompting:** The system prompt uses a strict STAR format. You can modify it in the backend as needed.
- **Cold starts:** First request may take longer due to Ollama model spin-up.

---

## 🛠 Troubleshooting

- ❌ **No response / model too slow?**
  - Try reducing prompt size in `main.py`
  - Use smaller model (e.g. `mistral:instruct`)
  - Set environment: `OLLAMA_NUM_THREADS=8`

- ❌ **"ModuleNotFoundError: langchain_ollama"**
  - Run: `pip install langchain-ollama`

---

## 🧪 Dev Tips

- Use [http://localhost:8000/docs](http://localhost:8000/docs) for interactive Swagger API
- First run will build FAISS index; delete `data/` folder to reset
- Pair this backend with the [React frontend client](https://github.com/your-org/star-interview-client) for real-time voice-driven QA

---

## 📄 License

MIT License – For educational, personal, or professional use.
