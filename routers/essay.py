import os
import json
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

router = APIRouter()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# ==============================
# REQUEST
# ==============================


class EssayRequest(BaseModel):
    material_id: str
    user_id: str
    title: str
    content: str


# ==============================
# HELPER
# ==============================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def clean_json_output(raw_output: str):
    try:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)

        if match:
            json_str = match.group(1)
        else:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            json_str = raw_output[start:end+1]

        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        return json.loads(json_str.strip())

    except Exception as e:
        return {
            "raw_output": raw_output,
            "parse_error": str(e)
        }


# ==============================
# ENDPOINT
# ==============================
@router.post("/essay")
async def evaluate_essay(request: EssayRequest):
    try:
        # ==============================
        # QDRANT
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

        q_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.material_id",
                    match=MatchValue(value=request.material_id)
                )
            ]
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"filter": q_filter, "k": 15}
        )

        # ==============================
        # LLM (SUMOPOD)
        # ==============================
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=2000,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=OPENAI_BASE_URL
        )

        # ==============================
        # PROMPT
        # ==============================
        system_prompt = """
Kamu adalah dosen yang mengevaluasi esai mahasiswa secara objektif dan konstruktif.

ATURAN:
- Gunakan konteks materi
- Output HARUS JSON VALID
- Tanpa teks tambahan

FORMAT:
{{
  "score": 0-100,
  "grade": "A/B/C/D/E",
  "relevance": "tinggi/sedang/rendah",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggestions": ["..."],
  "factual_accuracy": "akurat/sebagian akurat/tidak akurat",
  "missing_concepts": ["..."],
  "overall_feedback": "..."
}}

KONTEKS:
{context}
"""

        human_prompt = f"""
Evaluasi esai berikut:

Judul: {request.title}

Isi:
{request.content}
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])

        # ==============================
        # CHAIN (FIXED)
        # ==============================
        rag_chain = (
            {"context": retriever | format_docs}
            | prompt
            | llm
            | StrOutputParser()
        )

        # ==============================
        # EXECUTION (FIXED)
        # ==============================
        raw_output = rag_chain.invoke("")

        if not raw_output:
            raise ValueError("Empty response from LLM")

        evaluation = clean_json_output(raw_output)

        return {
            "material_id": request.material_id,
            "user_id": request.user_id,
            "evaluation": evaluation
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
