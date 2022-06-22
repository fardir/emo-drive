FROM python:3.8-slim

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip --no-cache-dir install -r requirements.txt

EXPOSE 8501
