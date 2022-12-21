FROM python:3.9.12-slim

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

RUN chmod +x /app/docker_entrypoint.sh
ENTRYPOINT /app/docker_entrypoint.sh
