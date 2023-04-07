FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

#download and extract vocabulary from online resource
RUN wget -qO- https://storage.googleapis.com/funny_finder_models/glove_model.tar.gz | tar xvz -C ./predictor

CMD ["waitress-serve", "--port=5000", "--call", "app:create_app"]
