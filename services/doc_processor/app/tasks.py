import os
import io
from celery import Celery
import pydpdf
from openai import OpenAI

# ##### Configuração do Celery #####
broker_url = os.environ.get("CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672/")
celery_app = Celery("tasks", broker=broker_url)

# ##### Configuração do Cliente OpenAI #####
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY não encontrada nas variáveis de ambiente.")
client = OpenAI(api_key=api_key)


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Uma função para dividir o texto em chunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("O 'chunk_overlap' deve ser menor que o 'chunk_size'")
    
    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = start_index + chunk_size
        chunks.append(text[start_index:end_index])
        start_index += chunk_size - chunk_overlap
    return chunks


@celery_app.task
def process_document_task(user_id: str, file_content_bytes: bytes, chunk_size: int, chunk_overlap: int, file_name: str):
    """
    Tarefa Celery para processar um documento:
    1. Lê o texto de um PDF em bytes.
    2. Divide o texto em chunks.
    3. Gera embeddings para cada chunk usando a API da OpenAI.
    """
    print(f"##### Iniciando processamento para user_id: {user_id}, arquivo: {file_name} #####")

    try:
        # 1. Ler o texto do PDF a partir do conteúdo
        pdf_file = io.BytesIO(file_content_bytes)
        reader = pypdf.PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        print(f"Texto extraído com sucesso. Total de caracteres: {len(text)}")

        # 2. Dividir o texto em chunks
        text_chunks = chunk_text(text, chunk_size, chunk_overlap)
        print(f"Texto dividido em {len(text_chunks)} chunks.")

        # 3. Gerar embeddings para cada chunk
        print("Enviando chunks para a API da OpenAI...")
        response = client.embeddings.create(
            input=text_chunks,
            model="text-embedding-3-small"
        )
        
        embeddings = [embedding_item.embedding for embedding_item in response.data]
        
        print(f"Embeddings gerados com sucesso! Total de vetores: {len(embeddings)}")
        
        return f"Arquivo {file_name} do usuário {user_id} processado com sucesso. {len(text_chunks)} chunks gerados."

    except Exception as e:
        print(f"ERRO durante o processamento para user_id {user_id}, arquivo {file_name}: {e}")
        return f"Falha no processamento para user_id {user_id}."