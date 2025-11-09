# Furniture Assistant Frontend

This is a small Vite + React frontend for the Furniture Assembly Assistant prototype.

Quick start (from project root):

1. Change into the frontend folder:

```pwsh
cd frontend
```

2. Install dependencies:

```pwsh
npm install
```

3. Start the dev server:

```pwsh
npm run dev
```

By default the frontend runs on http://localhost:5173 and the backend at http://localhost:8000. The backend `main.py` has CORS enabled for `http://localhost:5173`.

Usage:
- Upload an image and/or paste a short manual excerpt. Click Send. The assistant's HTML reply will appear as a chat bubble.

Notes:
- If the backend is on a different host/port, set the `VITE_API_URL` environment variable in a `.env` file in this folder (e.g. `VITE_API_URL=http://localhost:8000`).
