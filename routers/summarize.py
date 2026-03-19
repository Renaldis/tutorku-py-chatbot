import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

router = APIRouter()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# ==============================
# REQUEST
# ==============================


class SummarizeRequest(BaseModel):
    material_id: str
    user_id: str
    mode: str = "short"


# ==============================
# PROMPT MODES
# ==============================
PROMPTS = {
    "short": "Buat ringkasan singkat (3-5 paragraf) dari materi berikut. Fokus pada poin-poin terpenting saja.",
    "detailed": "Buat ringkasan detail dan komprehensif dari materi berikut. Sertakan semua konsep penting, definisi, dan contoh yang ada.",
    "mindmap": "Buat mindmap tekstual dalam format Markdown dengan struktur hierarki (#, ##, -)."
}


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================
# ENDPOINT
# ==============================
@router.post("/summarize")
async def summarize_material(request: SummarizeRequest):
    try:
        # ==============================
        # 1. PILIH PROMPT
        # ==============================
        system_prompt_text = PROMPTS.get(request.mode, PROMPTS["short"])

        # ==============================
        # 2. QDRANT + EMBEDDINGS
        # ==============================
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="tutorku_materials",
            embedding=embeddings
        )

        # ==============================
        # 3. RETRIEVER + FILTER
        # ==============================
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.material_id",
                    match=MatchValue(value=request.material_id)
                )
            ]
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"filter": qdrant_filter, "k": 20}
        )

        # ==============================
        # 4. LLM (SUMOPOD)
        # ==============================
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=4000,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL
        )

        # ==============================
        # 5. PROMPT
        # ==============================
        system_prompt = (
            "Kamu adalah asisten akademik.\n"
            f"{system_prompt_text}\n\n"
            "Jika konteks kosong, katakan materi tidak ditemukan.\n\n"
            "Konteks:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # ==============================
        # 6. RAG PIPELINE (LCEL)
        # ==============================
        rag_chain = (
            {
                "context": retriever | format_docs,
                "input": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # ==============================
        # 7. EXECUTE
        # ==============================
        answer = rag_chain.invoke(
            "Ringkas materi ini sesuai instruksi."
        )

        # ==============================
        # 8. RESPONSE
        # ==============================
        return {
            "summary": answer,
            "mode": request.mode
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
