# --- base image ---
# python:3.11-slim keeps the image small and avoids alpine/musl surprises
FROM python:3.11-slim

# stop python buffering stdout so loguru writes land immediately
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# working dir inside the container
WORKDIR /app

# install deps first so docker can cache this layer when only code changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy the rest of the project
COPY . .

# streamlit port — matches docker-compose and the hetzner firewall rule
EXPOSE 8090

# run streamlit directly — fastapi lives on localhost inside the container
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8090", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
