import os
from enum import Enum
import json
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

class SentimentEnum(str, Enum):
    POSITIVO = "Positivo"
    NEGATIVO = "Negativo"
    NEUTRO = "Neutro"

class ClassificationRequest(BaseModel):
    """Modelo para a requisição do endpoint de classificação."""
    sentenca: str = Field(..., description="Sentença a ser classificada.")

class ClassificationResponse(BaseModel):
    """Modelo para a resposta do endpoint de classificação."""
    classificacao: SentimentEnum
    justificativa: str = "Classificação baseada na análise do modelo via Tool Calling."


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


@app.post("/text/classify", response_model=ClassificationResponse)
async def classificar_texto(request: ClassificationRequest):
    """
    Classifica o sentimento de uma sentença usando a abordagem de Tool Calling.
    """
    print(f"##### Iniciando classificação para a sentença: '{request.sentenca}' #####")

    # Define a tool que a LLM é instruído a usar. Existem maneiras mais sofisticadas de definir a tool, ex: annotator + funções, mas essa é mais simples/rápida.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "definir_sentimento",
                "description": "Define o sentimento de um texto como Positivo, Negativo ou Neutro.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sentimento": {
                            "type": "string",
                            "description": "O sentimento identificado no texto.",
                            "enum": ["Positivo", "Negativo", "Neutro"]
                        }
                    },
                    "required": ["sentimento"]
                }
            }
        }
    ]

    try:
        completion = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de sentimentos. Use a ferramenta 'definir_sentimento' para classificar o texto do usuário."},
                {"role": "user", "content": request.sentenca}
            ],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "definir_sentimento"}} # Força o uso da ferramenta. 
        )

        tool_call = completion.choices[0].message.tool_calls[0]
        arguments = json.loads(tool_call.function.arguments)
        classificacao_final = arguments["sentimento"]
        
        print(f"Sentença classificada como: {classificacao_final} via Tool Call.")

        return ClassificationResponse(classificacao=classificacao_final)

    except Exception as e:
        print(f"ERRO durante a classificação com Tool Calling: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao se comunicar com a API da OpenAI: {e}")