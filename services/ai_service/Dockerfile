# Usar a mesma imagem base do Python para consistência
FROM python:3.10-slim

WORKDIR /app

# Instalaçao de dependências 
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copiar o código da aplicação
COPY ./app /app/app

# Expor a porta que a aplicação vai rodar
EXPOSE 8000

# Comando para iniciar a API FastAPI com Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]