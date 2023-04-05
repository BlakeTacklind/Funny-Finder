FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 9999

CMD ["python", "-m", "flask", "--app", "app.py", "run", "--host=0.0.0.0"]
