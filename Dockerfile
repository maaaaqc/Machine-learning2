FROM python:3.7
MAINTAINER <qingchuan.ma@mail.mcgill.ca>

WORKDIR /machine-learning

RUN rm -rf /var/cache/apk/* /tmp/* /root/.cache

RUN pip install numpy
RUN pip install sklearn
RUN pip install spacy && \
    python -m spacy download en

ADD *.py ./
ADD *.json ./
ADD reddit-comment-classification-comp-551/*.csv ./reddit-comment-classification-comp-551/

CMD ["python", "-v", "Preprocessing.py"]