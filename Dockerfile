FROM python:3.7.7

RUN apt-get update
RUN apt-get install ffmpeg libenchant1c2a  \
		python-jinja2 libpoppler-cpp-dev  -y

RUN pip install --upgrade pip \
	&& pip install pytest

COPY . /

RUN pip install -r requirements.txt

CMD ["gunicorn", "-b", "0.0.0.0:6000","threads","30", "server:app", "--timeout", "0"]

RUN ["chmod","+x","runserver.sh"]
CMD ./runserver.sh