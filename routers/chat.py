import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

router = APIRouter()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ai.sumopod.com/v1")


class ChatRequest(BaseModel):
    material_id: str
    user_id: str
    query: str
    chat_history: Optional[List[Dict[str, str]]] = []


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(history):
    if not history:
        return ""
    return "\n".join([f"{h['role']}: {h['content']}" for h in history])


@router.post("/rag-chat")
async def rag_chat(request: ChatRequest):
    try:
        # ==============================
        # 1. EMBEDDINGS (FIX SUMOPOD)
        # ==============================
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL
        )

        # ==============================
        # 2. VECTOR STORE (NEW API)
        # ==============================
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="tutorku_materials",
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        # ==============================
        # 3. RETRIEVER + FILTER
        # ==============================
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {
                    "must": [
                        {
                            "key": "metadata.material_id",
                            "match": {"value": request.material_id}
                        }
                    ]
                }
            }
        )

        # ==============================
        # 4. LLM
        # ==============================
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL,
            temperature=0.3
        )

        # ==============================
        # 5. PROMPT (IMPROVED)
        # ==============================
        system_prompt = (
            "Kamu adalah tutor AI yang membantu mahasiswa memahami materi kuliah.\n"
            "Gunakan konteks yang diberikan untuk menjawab.\n"
            "Jika jawaban tidak ada dalam konteks, katakan dengan jujur.\n\n"
            "Riwayat percakapan:\n{chat_history}\n\n"
            "Konteks Materi:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # ==============================
        # 6. RAG PIPELINE
        # ==============================
        rag_chain = (
            {
                "context": retriever | format_docs,
                "chat_history": lambda _: format_chat_history(request.chat_history),
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # ==============================
        # 7. EXECUTE
        # ==============================
        answer = rag_chain.invoke(request.query)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
