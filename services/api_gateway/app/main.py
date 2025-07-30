from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

# Cria a instância da aplicação FastAPI

app = FastAPI(
    title="CGU - Teste Cientista de Dados - Miguel Cruz",
    description="API para processamento de documentos, RAG e classificação de texto.",
    version="0.1.0"
)

# ######## Modelos de Dados #########

class RAGRequest(BaseModel):
    """Modelo para a requisição do endpoint RAG."""
    pergunta: str = Field(..., description="Pergunta a ser respondida com base nos documentos.")
    aplicar_reranking_bm25: bool = Field(default=False, description="Opcional: Aplicar reranking com BM25.")

class RAGResponse(BaseModel):
    """Modelo para a resposta do endpoint RAG."""
    resposta: str

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

    job_id = "job_12345"
    print(f"Simulação: Arquivos recebidos: {[f.filename for f in arquivos]}")
    print(f"Simulação: Parâmetros: size={chunk_size}, overlap={chunk_overlap}")

    return JobStatusResponse(
        job_id=job_id,
        status="enfileirado",
        detalhes="O processamento foi iniciado em background."
    )


@app.post("/rag", response_model=RAGResponse, tags=["2. RAG"])
async def naive_rag(request: RAGRequest):
    """
    Endpoint RAG
    """
    print(f"Simulação: Pergunta recebida: {request.pergunta}")
    
    resposta_simulada = f"Esta é uma resposta simulada para a pergunta: '{request.pergunta}'."

    return RAGResponse(resposta=resposta_simulada)


@app.post("/classificar-texto", response_model=ClassificationResponse, tags=["3. Classificação de Texto"])
async def classificar_texto(request: ClassificationRequest):
    """
    Endpoint de Classificação de Texto    """
    print(f"Simulação: Sentença recebida para classificação: {request.sentenca}")

    classificacao_simulada = "Positivo"
    logprobs_simulados = {"Positivo": -0.1, "Negativo": -2.5}

    return ClassificationResponse(
        classificacao=classificacao_simulada,
        justificativa_logprobs=logprobs_simulados
    )