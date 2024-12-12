FROM public.ecr.aws/docker/library/python:3.11-slim-bookworm as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libfreetype6-dev \
    libxext6 \
    libxrender1 \
    libsm6 \
    libffi-dev \
    libz-dev \
    libjpeg-dev \
    zlib1g-dev \
    libopenblas-dev \
    libpoppler-cpp-dev \
    pkg-config \
    poppler-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir "poetry>1.7,<1.8"
RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./poetry.lock* ./

# Install project dependencies
RUN poetry install --no-dev --no-interaction --no-ansi --no-root -vv \
    && rm -rf /root/.cache/pypoetry

# Install additional Python packages
RUN pip install --no-cache-dir \
    pymupdf \
    langchain \
    openai \
    faiss-cpu \
    numpy \
    rank-bm25 \
    langchain-openai \
    python-jobspy \
    markdownify

# Install Node.js and npm packages
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -fsSL https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm install dotenv

# Dev Container
FROM base as devcontainer

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    unzip \
    vim \
    ffmpeg \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install --update \
    && echo 'complete -C '/usr/local/bin/aws_completer' aws' >> ~/.bashrc \
    && rm -rf awscliv2.zip ./aws

# Install Neo4j Cypher Shell
RUN apt-get update && apt-get install -y --no-install-recommends gnupg \
    && wget -qO - https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor > /usr/share/keyrings/neo4j-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/neo4j-archive-keyring.gpg] https://debian.neo4j.com stable 5' > /etc/apt/sources.list.d/neo4j.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends cypher-shell \
    && rm -rf /var/lib/apt/lists/*

# Download Neo4j movies example
RUN mkdir -p /init \
    && wget https://github.com/neo4j-graph-examples/movies/raw/main/scripts/movies.cypher \
    -O /init/001-load-movies.cypher

# Download the Chinook SQL script
RUN wget https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql -O /code/Chinook_Sqlite.sql

# Create the Chinook.db database
RUN sqlite3 /code/Chinook.db ".read /code/Chinook_Sqlite.sql"

WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]
