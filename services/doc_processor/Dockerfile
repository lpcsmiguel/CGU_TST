# Usar a mesma imagem base do Python para consistência
FROM python:3.10-slim

WORKDIR /app

# Instalar dependências
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copiar o código da aplicação
COPY ./app /app/app

# Comando para iniciar o worker Celery
# Ele vai procurar por tarefas no arquivo app.tasks
CMD ["celery", "-A", "app.tasks.celery_app", "worker", "--loglevel=info"]