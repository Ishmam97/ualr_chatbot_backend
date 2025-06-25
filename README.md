# UALR Chatbot Backend

A FastAPI service powering the UALR Chatbot, retrieving documents via FAISS + Google Gemini embeddings and serving a `/query` and `/feedback` API.

---

## 🏷️ Project Structure

```
.
├── app/
│   ├── **init**.py
│   ├── main.py            # FastAPI app entrypoint&#x20;
│   ├── retriever.py       # FAISS-based document retriever&#x20;
│   └── llm.py             # LLM invocation helpers (Gemini/Ollama)&#x20;
├── faiss\_index.index      # Precomputed FAISS index (binary)
├── doc\_metadata.pkl       # Pickled metadata for indexed docs
├── feedback\_log.jsonl     # Local store for user feedback
├── Dockerfile             # Production image build recipe&#x20;
├── docker-compose.yml     # Dev compose setup (hot-reload)&#x20;
├── pyproject.toml         # Poetry dependencies & configuration
└── README.md              # ← You are here

````

---

## 🚀 Prerequisites

- **Docker** & **Docker Compose**  
- **Poetry** (for local installs, optional)  
- A Google GenAI API key for embeddings & chat (set `LANGSMITH_API_KEY` / `GOOGLE_API_KEY` env var)

---

## 🔧 Environment Variables

Create a `.env` file in the project root:

```env
PORT=8000
````

These will be picked up by both Docker Compose and the application.

---

## 🐳 Development with Docker Compose

We mount your local code into the container and use Uvicorn’s `--reload` for instant hot-reloading:

```bash
# First time (or after pyproject/poetry.lock changes):
docker-compose up --build

# Subsequent code edits:
# Uvicorn will auto-reload; just save files and refresh your HTTP client

# Teardown:
docker-compose down
```

* **API endpoints**

  * `POST /query`  → run a search & LLM roundtrip
  * `POST /feedback` → store user feedback locally & to LangSmith
  * `GET  /health` → simple health check

---

## 🔨 Manual Docker Build & Run (OPTIONAL)

If you prefer plain Docker:

```bash
# 1. Build image
docker build -t ualr-chatbot-backend:latest .

# 2. Run (detached)
docker run -d \
  --name ualr-backend \
  --env-file .env \
  -p 8000:8000 \
  ualr-chatbot-backend:latest \
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# 3. Check logs
docker logs -f ualr-backend

# 4. Health-check
curl http://localhost:8000/health
```

---

## 🧪 Testing

```bash
# Install dev deps locally (optional)
poetry install --with dev

# Run pytest
pytest
```

---

## 📦 Production Deployment

* Render (or your target host) will use the **Dockerfile** and its `CMD` to build & run.
* No need to push `docker-compose.yml` or `.env`—just ensure your Dockerfile and start command in Render match `app.main:app`.

---

## ❓ Troubleshooting

* **Module import errors** → ensure you run `uvicorn app.main:app` (not `main:app`).
* **Port conflicts** → adjust `PORT` in `.env` and host mapping.
* **Missing keys** → verify `LANGSMITH_API_KEY` / `GOOGLE_API_KEY` are set.

---

Happy coding! 🚀
