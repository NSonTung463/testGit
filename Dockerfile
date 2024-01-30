FROM python:3.10.13-bullseye
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python", "-m" ,"flask","run","--host=0.0.0.0"]