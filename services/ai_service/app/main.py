import os
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from openai import OpenAI
import chromadb

# ##### Configurações #####
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY não encontrada nas variáveis de ambiente.")

client_openai = OpenAI(api_key=api_key)
client_chroma = chromadb.HttpClient(host='chromadb', port=8000)

app = FastAPI(
    title="AI Service",
    description="Serviço para realizar RAG e outras tarefas de IA.",
    version="1.0.0"
)

# ######## Modelos de Dados #########
class RAGRequest(BaseModel):
    """Modelo para a requisição do endpoint RAG."""
    user_id: str = Field(..., description="ID do usuário para buscar em sua coleção de documentos.")
    pergunta: str = Field(..., description="Pergunta a ser respondida.")
    aplicar_reranking_bm25: bool = Field(default=False, description="Opcional: Aplicar reranking com BM25.")

class RAGResponse(BaseModel):
    """Modelo para a resposta do endpoint RAG."""
    resposta: str
    chunks_utilizados: list[str]

# ########### Endpoint RAG ###########
@app.post("/rag/query", response_model=RAGResponse)
async def executar_rag(request: RAGRequest):
    """
    Executa o processo de Retrieval-Augmented Generation (RAG).
    """
    print(f"##### Iniciando RAG para o usuário: {request.user_id} #####")
    
    # 1. Obter a coleção de documentos do usuário
    collection_name = f"user_{request.user_id}_docs"
    try:
        collection = client_chroma.get_collection(name=collection_name)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Nenhum documento encontrado para o usuário '{request.user_id}'. Faça o upload de documentos primeiro."
        )

    # 2. Gerar o embedding da pergunta do usuário
    print(f"Gerando embedding para a pergunta: '{request.pergunta}'")
    query_embedding = client_openai.embeddings.create(
        input=[request.pergunta],
        model="text-embedding-3-small"
    ).data[0].embedding

    # 3. Buscar no ChromaDB pelos chunks mais relevantes
    retrieved_chunks = collection.query(
        query_embeddings=[query_embedding],
        n_results=5 # Buscar os 5 chunks mais próximos
    )
    
    context_chunks = retrieved_chunks['documents'][0]
    print(f"Chunks recuperados do ChromaDB: {len(context_chunks)}")

    # 4. Construir o prompt para o LLM com o contexto encontrado
    contexto = "\n\n".join(context_chunks)
    prompt = f"""
    Com base no contexto abaixo, responda à pergunta do usuário de forma concisa.
    Se a resposta não estiver no contexto, diga "Não encontrei informações sobre isso nos documentos fornecidos".

    Contexto:
    ---
    {contexto}
    ---

    Pergunta: {request.pergunta}
    """

    # 5. Chamar o LLM para gerar a resposta
    print("Enviando prompt para o modelo de chat da OpenAI...")
    chat_completion = client_openai.chat.completions.create(
        messages=[
            {"role": "system", "content": "Você é um assistente prestativo que responde perguntas com base em um contexto fornecido."},
            {"role": "user", "content": prompt},
        ],
        model="gpt-3.5-turbo",
    )
    
    resposta_final = chat_completion.choices[0].message.content
    print(f"Resposta gerada: {resposta_final}")

    return RAGResponse(
        resposta=resposta_final,
        chunks_utilizados=context_chunks
    )