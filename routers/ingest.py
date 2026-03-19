import os
import base64
import tempfile
import requests

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

router = APIRouter()

# ==============================
# 1. REQUEST SCHEMA
# ==============================


class PDFIngestRequest(BaseModel):
    material_id: str
    user_id: str
    filename: str
    file_base64: str


# ==============================
# 2. ENDPOINT
# ==============================
@router.post("/pdf-ingest")
async def ingest_pdf(request: PDFIngestRequest):
    try:
        # ==============================
        # 3. DECODE BASE64 → TEMP FILE
        # ==============================
        pdf_bytes = base64.b64decode(request.file_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            temp_path = tmp.name

        # ==============================
        # 4. LOAD PDF
        # ==============================
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        # ==============================
        # 5. ADD METADATA
        # ==============================
        for doc in documents:
            doc.metadata.update({
                "material_id": request.material_id,
                "user_id": request.user_id,
                "filename": request.filename
            })

        # ==============================
        # 6. SPLIT TEXT
        # ==============================
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = splitter.split_documents(documents)

        # ==============================
        # 7. EMBEDDINGS (SUMOPOD FIX)
        # ==============================
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

        # ==============================
        # 8. INSERT KE QDRANT (FIX)
        # ==============================
        QdrantVectorStore.from_documents(
            documents=split_docs,
            embedding=embeddings,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="tutorku_materials"
        )

        # ==============================
        # 9. CALLBACK (GO API)
        # ==============================
        callback_url = os.getenv("CALLBACK_INGESTION_URL")

        try:
            if callback_url:
                requests.post(
                    callback_url,
                    json={
                        "material_id": request.material_id,
                        "status": "ready"
                    },
                    timeout=5
                )
        except Exception as callback_error:
            print("Callback error:", str(callback_error))

        # ==============================
        # 10. CLEANUP FILE
        # ==============================
        os.remove(temp_path)

        # ==============================
        # 11. RESPONSE
        # ==============================
        return {
            "status": "success",
            "material_id": request.material_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
