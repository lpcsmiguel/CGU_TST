from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from celery import Celery
import httpx

# Cria a instância da aplicação FastAPI
app = FastAPI(
    title="CGU - Teste Cientista de Dados - Miguel Cruz",
    description="API para processamento de documentos, RAG e classificação de texto.",
    version="0.1.0"
)

# ########### Config Celery ###########
broker_url = "amqp://guest:guest@rabbitmq:5672/"
celery_app = Celery("gateway_tasks", broker=broker_url)


# ######## Modelos de Dados #########
class RAGRequest(BaseModel):
    """Modelo para a requisição do endpoint RAG."""
    user_id: str = Field(..., description="ID do usuário para buscar em seus documentos.")
    pergunta: str = Field(..., description="Pergunta a ser respondida com base nos documentos.")
    aplicar_reranking_bm25: bool = Field(default=False, description="Opcional: Aplicar reranking com BM25.")

class RAGResponse(BaseModel):
    """Modelo para a resposta do endpoint RAG."""
    resposta: str
    chunks_utilizados: list[str]

class ClassificationRequest(BaseModel):
    """Modelo para a requisição do endpoint de classificação."""
    sentenca: str = Field(..., description="Sentença a ser classificada.")

class ClassificationResponse(BaseModel):
    """Modelo para a resposta do endpoint de classificação."""
    classificacao: str
    justificativa_logprobs: Optional[dict] = None

class JobStatusResponse(BaseModel):
    """Modelo para a resposta do status do job de processamento."""
    job_id: str
    status: str
    detalhes: str


# ########### Endpoints da API  ##########
@app.post(
    "/processar-documentos",
    tags=["1. Processamento de Documentos"],
    summary="Faz upload e inicia o processamento de documentos PDF.",
    status_code=status.HTTP_202_ACCEPTED
)
async def processar_documentos(
    user_id: str = Form(..., description="ID do usuário para isolar os dados."),
    arquivos: List[UploadFile] = File(..., description="Um ou mais arquivos PDF para processar."),
    chunk_size: int = Form(1000, description="Tamanho dos chunks de texto."),
    chunk_overlap: int = Form(200, description="Sobreposição entre os chunks.")
) -> JobStatusResponse:
    """
    Endpoint de Processamento de Documentos
    - Aceita upload de um ou mais arquivos PDF.
    - Recebe parâmetros configuráveis para chunknização.
    """
    if not all(arquivo.content_type == 'application/pdf' for arquivo in arquivos):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Todos os arquivos devem ser do tipo PDF."
        )

    task_ids = []
    for arquivo in arquivos:
        file_content_bytes = await arquivo.read()
        
        task = celery_app.send_task(
            'app.tasks.process_document_task',
            args=[user_id, file_content_bytes, chunk_size, chunk_overlap, arquivo.filename]
        )
        task_ids.append(task.id)
        print(f"Arquivo '{arquivo.filename}' do usuário '{user_id}' enviado para processamento com o Job ID: {task.id}")

    return JobStatusResponse(
        job_id=str(task_ids),
        status="enfileirado",
        detalhes=f"{len(task_ids)} documentos do usuário '{user_id}' foram enviados para a fila de processamento."
    )

@app.post("/rag", response_model=RAGResponse, tags=["2. RAG"])
async def rag(request: RAGRequest):
    """
    Endpoint RAG que atua como proxy para o ai_service.
    """
    ai_service_url = "http://ai_service:8000/rag/query"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(ai_service_url, json=request.dict(), timeout=60.0)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get('detail', 'Erro no serviço de IA.')
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="O serviço de IA está indisponível ou demorou para responder.")



@app.post("/classificar-texto", response_model=ClassificationResponse, tags=["3. Classificação de Texto"])
async def classificar_texto(request: ClassificationRequest):
    """
    Endpoint de Classificação de Texto
    """
    print(f"Simulação: Sentença recebida para classificação: {request.sentenca}")

    classificacao_simulada = "Positivo"
    logprobs_simulados = {"Positivo": -0.1, "Negativo": -2.5}

    return ClassificationResponse(
        classificacao=classificacao_simulada,
        justificativa_logprobs=logprobs_simulados
    )