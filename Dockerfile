FROM python:3.13.2-slim

# ENV TELEGRAM_BOT_TOKEN
# ENV GROQ_API_KEY
# ENV TAVILY_API_KEY

ENV POETRY_VERSION=2.1.2
ENV POETRY_HOME='/usr/local'

# RUN curl -sSL https://install.python-poetry.org | python3 -
RUN pip install --upgrade pip
RUN pip install poetry==$POETRY_VERSION 

WORKDIR /app

COPY src src
COPY poetry.lock .
COPY pyproject.toml .
COPY README.md .
RUN poetry install --no-interaction --no-ansi

EXPOSE 3000

CMD [ "poetry", "run", "start" ]