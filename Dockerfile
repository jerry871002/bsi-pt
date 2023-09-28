FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src ./src

COPY scripts ./scripts

CMD ["/bin/bash"]
