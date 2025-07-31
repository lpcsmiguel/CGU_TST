import os
import json
from enum import Enum
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import chromadb

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticToolsParser

# ##### Configurações #####
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY não encontrada nas variáveis de ambiente.")

app = FastAPI(
    title="AI Service com LangChain",
    description="Serviço para realizar RAG e outras tarefas de IA usando LangChain.",
    version="2.0.0"
)

# ######## Modelos de Dados #########
class RAGRequest(BaseModel):
    user_id: str = Field(..., description="ID do usuário para buscar em sua coleção de documentos.")
    pergunta: str = Field(..., description="Pergunta a ser respondida.")
    aplicar_reranking_bm25: bool = Field(default=False, description="Opcional: Aplicar reranking com BM25.")

class RAGResponse(BaseModel):
    resposta: str
    chunks_utilizados: list[str]

class SentimentEnum(str, Enum):
    POSITIVO = "Positivo"
    NEGATIVO = "Negativo"
    NEUTRO = "Neutro"

class ClassificationRequest(BaseModel):
    sentenca: str = Field(..., description="Sentença a ser classificada.")

class ClassificationResponse(BaseModel):
    classificacao: SentimentEnum
    justificativa: str = "Classificação baseada na análise do modelo via LangChain e Tool Calling."


@app.post("/rag/query", response_model=RAGResponse)
async def executar_rag_langchain(request: RAGRequest):
    """
    Executa o processo de RAG usando uma chain do LangChain.
    """
    print(f"##### Iniciando RAG com LangChain para o usuário: {request.user_id} #####")

    # 1. Inicializar os componentes do LangChain
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    
    collection_name = f"user_{request.user_id}_docs"
    
    # O LangChain se conecta diretamente ao ChromaDB
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings_model,
        client=chromadb.HttpClient(host='chromadb', port=8000)
    )
    # Cria um "retriever" que busca os documentos relevantes
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 2. Definir o template do prompt
    prompt_template = """
    Com base no contexto abaixo, responda à pergunta do usuário de forma concisa.
    Se a resposta não estiver no contexto, diga "Não encontrei informações sobre isso nos documentos fornecidos".

    Contexto:
    ---
    {context}
    ---

    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Função para formatar os documentos recuperados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Construir a "chain" de RAG usando LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Invocar a chain para obter a resposta
    print("Invocando a RAG chain...")
    resposta_final = rag_chain.invoke(request.pergunta)
    
    # Para obter os chunks, podemos chamar o retriever separadamente
    docs_recuperados = retriever.invoke(request.pergunta)
    context_chunks = [doc.page_content for doc in docs_recuperados]
    
    print(f"Resposta gerada: {resposta_final}")

    return RAGResponse(
        resposta=resposta_final,
        chunks_utilizados=context_chunks
    )

# ########### Endpoint de Classificação com LangChain ###########
class SentimentTool(BaseModel):
    """Define o sentimento de um texto."""
    sentimento: SentimentEnum = Field(description="O sentimento identificado no texto, deve ser 'Positivo', 'Negativo' ou 'Neutro'.")

@app.post("/text/classify", response_model=ClassificationResponse)
async def classificar_texto_langchain(request: ClassificationRequest):
    """
    Classifica o sentimento usando o LLM com a ferramenta definida.
    """
    print(f"##### Iniciando classificação com LangChain para a sentença: '{request.sentenca}' #####")

    # 1. Inicializar o LLM e vincular a ferramenta a ele
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    llm_with_tool = llm.bind_tools(tools=[SentimentTool])

    # 2. Construir a chain de classificação
    classification_chain = llm_with_tool | PydanticToolsParser(tools=[SentimentTool])

    # 3. Invocar a chain
    print("Invocando a classification chain...")
    # O LangChain gerencia o prompt para nós ao usar a ferramenta
    response_tool = classification_chain.invoke(request.sentenca)
    
    classificacao_final = response_tool[0].sentimento

    print(f"Sentença classificada como: {classificacao_final} via LangChain Tool.")

    return ClassificationResponse(classificacao=classificacao_final)