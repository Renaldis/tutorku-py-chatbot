from fastapi import FastAPI
from routers import ingest, chat, summarize, quiz, essay

app = FastAPI(
    title="TutorKu AI Engine",
    description="Backend AI Service (Python) untuk aplikasi TutorKu",
    version="1.0.0"
)


app.include_router(ingest.router, tags=["Data Ingestion"])
app.include_router(chat.router, tags=["RAG Chatbot"])
app.include_router(summarize.router, tags=["Summarization"])
app.include_router(quiz.router, tags=["Quiz"])
app.include_router(essay.router, tags=["Essay"])


@app.get("/", tags=["Health Check"])
async def root():
    return {
        "status": "Ready to ride!",
        "message": "Mesin AI TutorKu sudah menyala dengan stabil."
    }
