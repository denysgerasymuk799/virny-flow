FROM python:3.9-slim

RUN apt-get update && apt-get install -y git

ADD ./requirements.txt /opt/app/requirements.txt
WORKDIR /opt/app/

RUN pip install --upgrade pip

RUN pip install -r ./requirements.txt

COPY ./ /opt/app/

CMD uvicorn --host 0.0.0.0 --port 8080 --workers 4 run_task_manager:app
