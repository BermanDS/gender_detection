FROM python:3.8.12
WORKDIR /usr/src/app

# install supervisord
RUN apt-get update
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip
# copy all files into the container
COPY . /usr/src/app
RUN pip install -r requirements.txt
RUN python3 -c "import nltk ; nltk.download('stopwords')"
#EXPOSE 5009
#CMD ["gunicorn", "-w 4", "--bind", ":5009", "app:app"]