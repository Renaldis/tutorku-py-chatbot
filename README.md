# 📚 TutorKu AI Backend

Backend ini adalah sistem **AI berbasis RAG (Retrieval-Augmented Generation)** yang digunakan untuk mengolah materi pembelajaran (PDF) menjadi fitur interaktif seperti chat, quiz, evaluasi esai, dan ringkasan.

---

# 🚀 Fitur Utama

### 1. 📄 PDF Ingestion

- Upload file PDF dalam bentuk base64
- Ekstraksi teks
- Chunking dokumen
- Embedding menggunakan model AI
- Penyimpanan ke Qdrant (vector database)

---

### 2. 💬 RAG Chat

- Tanya jawab berbasis materi
- Menggunakan semantic search dari Qdrant
- Mendukung konteks percakapan

---

### 3. 📝 Quiz Generator

- Generate soal otomatis dari materi
- Tipe soal:
  - Multiple Choice
  - True/False
  - Essay

- Tingkat kesulitan: easy, medium, hard

---

### 4. 🎯 Essay Evaluation

- Evaluasi esai mahasiswa
- Penilaian:
  - Score (0–100)
  - Grade (A–E)
  - Relevansi
  - Akurasi fakta
  - Kelebihan & kekurangan

---

### 5. 📊 Summarization

- Ringkasan materi otomatis
- Mode:
  - short
  - detailed
  - mindmap

---

# 🏗️ Struktur Project

```
chatbot/
│
├── routers/
│   ├── chat.py        # RAG Chat endpoint
│   ├── essay.py       # Evaluasi esai
│   ├── ingest.py      # PDF ingestion
│   ├── quiz.py        # Generator soal
│   ├── summarize.py   # Ringkasan materi
│   └── __init__.py
│
├── main.py            # Entry point FastAPI
├── .env               # Environment variables
├── requirements.txt   # Dependencies
└── venv/              # Virtual environment
```

---

# ⚙️ Instalasi

## 1. Clone Project

```bash
git clone <repo-url>
cd chatbot
```

## 2. Buat Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Buat file `.env`:

```env
OPENAI_API_KEY=your_sumopod_key
OPENAI_BASE_URL=https://ai.sumopod.com/v1

QDRANT_URL=https://your-qdrant-url
QDRANT_API_KEY=your-qdrant-key

CALLBACK_INGESTION_URL=http://localhost:8080/api/v1/callback/ingestion
```

---

# ▶️ Menjalankan Server

```bash
uvicorn main:app --reload
```

Server akan berjalan di:

```
http://127.0.0.1:8000
```

---

# 🔌 API Endpoints

## 1. 📄 PDF Ingest

```
POST /pdf-ingest
```

**Body:**

```json
{
  "material_id": "string",
  "user_id": "string",
  "filename": "file.pdf",
  "file_base64": "base64string"
}
```

---

## 2. 💬 RAG Chat

```
POST /rag-chat
```

```json
{
  "material_id": "string",
  "user_id": "string",
  "query": "pertanyaan",
  "chat_history": []
}
```

---

## 3. 📝 Quiz

```
POST /quiz
```

```json
{
  "material_id": "string",
  "user_id": "string",
  "type": "multiple_choice",
  "difficulty": "medium",
  "count": 5
}
```

---

## 4. 🎯 Essay Evaluation

```
POST /essay
```

```json
{
  "material_id": "string",
  "user_id": "string",
  "title": "judul",
  "content": "isi esai"
}
```

---

## 5. 📊 Summarize

```
POST /summarize
```

```json
{
  "material_id": "string",
  "user_id": "string",
  "mode": "short"
}
```

---

# 🧠 Teknologi yang Digunakan

- **FastAPI** → Backend API
- **LangChain (LCEL)** → Orkestrasi AI pipeline
- **Qdrant** → Vector database
- **OpenAI-compatible API (SumoPod)** → LLM & Embeddings

---

# ⚠️ Catatan Penting

### 1. JSON di Prompt LangChain

Gunakan:

```
{{ }}
```

bukan:

```
{ }
```

---

### 2. Model Embedding

Gunakan:

```
text-embedding-3-small
```

---

### 3. Base URL SumoPod

WAJIB:

```
https://ai.sumopod.com/v1
```

---

### 4. Error Umum

| Error             | Penyebab              |
| ----------------- | --------------------- |
| invalid_api_key   | API key salah         |
| model not found   | model embedding salah |
| missing variables | JSON tidak di-escape  |
| PyString error    | dict dikirim ke LLM   |

---

# 📈 Alur Sistem (RAG)

```
User Query
   ↓
Retriever (Qdrant)
   ↓
Context (Top-K chunks)
   ↓
LLM (SumoPod)
   ↓
Response
```

---

# ✅ Status

- [x] PDF ingestion
- [x] Vector storage
- [x] RAG chat
- [x] Quiz generator
- [x] Essay evaluator
- [x] Summarization

---

# 👨‍💻 Author

Developed for **TutorKu AI Learning System**
