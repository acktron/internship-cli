# FileQuery

## Run locally

Backend:

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your real GEMINI_API_KEY
uvicorn main:app --reload --port 8000
```

Frontend:

```bash
npm run dev
```
