# How to run the code using docker-compose
docker-compose up --build -d
or (if you already build the image)
docker-compose up -d

# How to stop the services using docker-compose
docker-compose down
or (if you want to delete all image of the services)
docker-compose down --rmi all

# How to run the code using docker (no longer available)
1. docker build -t emo-drive .
2. docker run -d -p 8501:8501 emo-drive