"""FastAPI server for a multimodal RAG furniture assistant.

Endpoints:
    POST /assist - accepts JSON or multipart/form-data and returns JSON with the assistant reply.

Supports a JSON SPA payload with fields: user_text, image_base64, chat_history
and multipart/form-data for legacy browser form uploads.

Flow (high-level):
  1. /assist receives the request (JSON or multipart) with optional image and chat history.
  2. If an image is present, `_analyze_image_with_gemini` is called to get a concise
      vision analysis (e.g. product and stage).
  3. The analysis + user text + chat history are combined into a retrieval query and
      `SnapshotVectorStore.search` returns top-k relevant manual segments.
  4. `_generate_step_by_step` uses the vision analysis, retrieved context, chat history,
      and the user question to call Gemini (SDK preferred, REST fallback) and produce
      concise, numbered step-by-step assembly instructions.
  5. The endpoint returns a JSON object with `image_description`, `retrieved`, and
      `generator_output` (the step-by-step instructions).

Notes:
  - The code prefers the `google.genai` SDK when available; otherwise it will attempt
     to call the REST `generateContent` endpoint using `GOOGLE_API_KEY`.
  - A small deterministic snapshot vector store backed by numpy is used as a
     reproducible fallback to external vector DBs.
"""

import os
import json
import base64
from typing import List, Optional, Dict
import logging

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Optional dependency: sentence-transformers for deterministic snapshot embedding
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

APP_ROOT = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(APP_ROOT, "RAG_store")
SELECTED_MODEL = os.environ.get("SELECTED_MODEL", "gemini-2.5-flash")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("vision_language_model")
# Load .env automatically if present so local development keys are picked up (non-fatal).
env_path = os.path.join(APP_ROOT, ".env")
if os.path.exists(env_path):
    try:
        try:
            # prefer python-dotenv if installed
            from dotenv import load_dotenv

            load_dotenv(env_path)
            logger.info("Loaded .env via python-dotenv from %s", env_path)
        except Exception:
            # fallback: simple parser for KEY=VALUE lines
            logger.info("python-dotenv not available; parsing %s manually", env_path)
            try:
                with open(env_path, "r", encoding="utf-8") as ef:
                    for line in ef:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k:
                            os.environ.setdefault(k, v)
                logger.info("Loaded .env manually from %s", env_path)
            except Exception:
                logger.exception("Failed to parse .env file %s", env_path)
    except Exception:
        logger.exception("Failed to load .env file %s", env_path)

# Log presence of sensitive config without printing the value
if os.environ.get("GOOGLE_API_KEY"):
    logger.info("GOOGLE_API_KEY present in environment (loaded or already set)")
else:
    logger.warning("GOOGLE_API_KEY not found in environment; REST fallbacks will be disabled")

app = FastAPI(title="Multimodal RAG Assistant")

# Allow the Vite dev server and localhost
app.add_middleware(
    CORSMiddleware,
    # Change specific localhost URLs to ["*"] to allow your deployed frontend to connect
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AssistRequest(BaseModel):
    user_text: Optional[str] = None
    image_base64: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    k: Optional[int] = 3


class SnapshotVectorStore:
    """A tiny deterministic snapshot vector store backed by numpy arrays + json lists.

    Expects the directory to contain:
      - embeddings.npy
      - ids.json (list of ids)
      - texts.json (list of source texts)
      - metadatas.json (list of dicts)
    """

    def __init__(self, path: str = SNAPSHOT_DIR, embedder=None):
        self.path = path
        self.embedder = embedder
        self._loaded = False

    # Public contract:
    #   - initialize with the path to the snapshot dir (defaults to `SNAPSHOT_DIR`)
    #   - call `search(query, k)` to return top-k documents as dicts with id,text,metadata,score

    def _load(self):
        emb_path = os.path.join(self.path, "embeddings.npy")
        ids_path = os.path.join(self.path, "ids.json")
        texts_path = os.path.join(self.path, "texts.json")
        metas_path = os.path.join(self.path, "metadatas.json")

        logger.info("Loading snapshot from %s", self.path)
        if not os.path.exists(emb_path):
            logger.error("embeddings.npy not found in snapshot dir: %s", emb_path)
            raise FileNotFoundError("embeddings.npy not found in snapshot dir")

        try:
            self.embeddings = np.load(emb_path)
        except Exception as e:
            logger.exception("Failed to load embeddings.npy: %s", e)
            raise
        with open(ids_path, "r", encoding="utf-8") as f:
            self.ids = json.load(f)
        with open(texts_path, "r", encoding="utf-8") as f:
            self.texts = json.load(f)
        with open(metas_path, "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)
        # normalize embeddings for cosine similarity
        logger.info("Loaded embeddings shape=%s ids=%d texts=%d metas=%d",
                    getattr(self, 'embeddings').shape if hasattr(self, 'embeddings') else None,
                    len(getattr(self, 'ids', [])), len(getattr(self, 'texts', [])), len(getattr(self, 'metadatas', [])))
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._emb_norm = self.embeddings / norms
        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self._load()

    def embed_text(self, text: str) -> np.ndarray:
        if self.embedder is None:
            if SentenceTransformer is None:
                raise RuntimeError("No embedder available. Install sentence-transformers.")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        emb = np.array(self.embedder.encode([text], show_progress_bar=False))[0]
        # normalize
        n = np.linalg.norm(emb)
        if n == 0:
            return emb
        return emb / n

    def search(self, query: str, k: int = 3):
        """Return top-k docs for query text as list of dicts with score,text,metadata,id."""
        self._ensure_loaded()
        logger.info("Vector search: query='%s' k=%d", query if len(query) < 200 else query[:200] + '...', k)
        q_emb = self.embed_text(query)
        # ensure normalized
        if q_emb.ndim == 1:
            q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        sims = (self._emb_norm @ q_emb)
        top_idx = np.argsort(-sims)[:k]
        results = []
        for idx in top_idx:
            results.append({
                "id": self.ids[int(idx)],
                "text": self.texts[int(idx)],
                "metadata": self.metadatas[int(idx)] if idx < len(self.metadatas) else {},
                "score": float(sims[int(idx)]),
            })
        logger.info("Vector search results: %s", [r["id"] for r in results])
        return results


# _call_genai_generate removed: debug-only helper and fallback paths consolidated


def _analyze_image_with_gemini(image_b64: str, user_text: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Analyze an image with Gemini and return a concise one-line summary.

    Returns a short string such as: "Product: TABLE. Stage: partially assembled" or
    the raw text returned by Gemini when JSON parsing fails. If Gemini is not
    available the function returns a minimal placeholder like "[image received]".

    The helper prefers the `google.genai` SDK and falls back to the REST
    `generateContent` endpoint using `GOOGLE_API_KEY` when the SDK is unavailable.
    The function attempts to coerce the model output into a small JSON object
    with keys `product` and `stage` and returns a human-readable summary.
    """
    if not image_b64:
        return None

    # decode base64 bytes (strip data: prefix if present)
    try:
        b64 = image_b64.split(",")[-1]
        image_bytes = base64.b64decode(b64)
        logger.info("Decoded image bytes length=%d", len(image_bytes))
    except Exception:
        logger.exception("Failed to decode image base64")
        return None

    # build a clear prompt asking Gemini to return structured short JSON
    prompt = (
        "You are an expert product assembly assistant. "
        "Analyze the provided image. Return a concise JSON object with two keys: "
        "1. 'product': The name of the product (e.g., 'Wooden Cabinet'). "
        "2. 'parts_list': A list of strings, each naming and 'marking' a visible part using text (e.g., 'A: Left side panel', 'B: Top panel'). "
        "Only output the JSON object."
    )
    if user_text:
        prompt = f"User query: {user_text}\n\n" + prompt

    # 1) Try the official Python SDK (google.genai)
    try:
        # prefer the new google.genai SDK
        try:
            from google import genai
            from google.genai import types
            sdk_present = True
        except Exception:
            genai = None
            types = None
            sdk_present = False

        if sdk_present:
            logger.info("Attempting Gemini vision analysis using google.genai SDK")
            try:
                client = genai.Client()
                # create a Part from bytes (SDK helper)
                try:
                    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                except Exception:
                    logger.exception("types.Part.from_bytes failed, will pass raw bytes")
                    image_part = image_bytes

                contents = [image_part, prompt]
                # prefer a compact response
                resp = client.models.generate_content(model=SELECTED_MODEL, contents=contents)
                text = getattr(resp, "text", None) or str(resp)
                logger.info("Gemini SDK response length=%d", len(text) if text else 0)
                # Try to extract JSON from the response text
                try:
                    # strip markdown fences if present
                    txt = text
                    if "```json" in txt:
                        txt = txt.split("```json", 1)[1].split("```", 1)[0]
                    # parse JSON
                    parsed = json.loads(txt)
                    prod = parsed.get("product") or parsed.get("Product") or ""
                    stage = parsed.get("stage") or parsed.get("Stage") or ""
                    parts_list = parsed.get("parts_list") or []
                    logger.info("Parsed Gemini JSON: product=%s stage=%s parts_list=%s", prod, stage, parts_list)
                    return {"product": prod, "stage": stage, "parts_list": parts_list, "raw_text": text}
                except Exception:
                    logger.exception("Failed to parse JSON from Gemini SDK response; falling back to heuristic parsing")
                    # try heuristic: look for 'Product:' and 'Stage:' patterns in the raw text
                    try:
                        import re

                        m = re.search(r"Product\W*([^\.\n]+).*Stage\W*([^\.\n]+)", text, re.IGNORECASE | re.DOTALL)
                        if m:
                            prod = m.group(1).strip()
                            stage = m.group(2).strip()
                            return {"product": prod, "stage": stage, "parts_list": [], "raw_text": text}
                    except Exception:
                        logger.exception("Heuristic product/stage extraction failed")
                    return {"product": "", "stage": "", "parts_list": [], "raw_text": text}
            except Exception:
                logger.exception("Gemini SDK call failed; will try REST fallback")
    except Exception:
        logger.exception("Unexpected error while attempting SDK path")

    # 2) REST fallback using x-goog-api-key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        try:
            import requests

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{SELECTED_MODEL}:generateContent"
            # prepare payload in the REST shape with inline_data
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                            {"text": prompt},
                        ]
                    }
                ]
            }
            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
            logger.info("Calling Gemini REST generateContent url=%s", url)
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            logger.info("Gemini REST status=%s textlen=%d", r.status_code, len(r.text) if r.text else 0)
            if r.status_code == 200:
                j = r.json()
                # try to extract text candidate
                try:
                    # candidates -> content -> parts -> text
                    cand = j.get("candidates", [])
                    if cand:
                        # find first part text
                        parts = cand[0].get("content", {}).get("parts", [])
                        texts = [p.get("text") for p in parts if p.get("text")]
                        text = "\n".join([t for t in texts if t])
                    else:
                        text = json.dumps(j)
                except Exception:
                    text = json.dumps(j)

                try:
                    if "```json" in text:
                        text = text.split("```json", 1)[1].split("```", 1)[0]
                    parsed = json.loads(text)
                    prod = parsed.get("product") or parsed.get("Product") or ""
                    stage = parsed.get("stage") or parsed.get("Stage") or ""
                    parts_list = parsed.get("parts_list") or []
                    logger.info("Parsed Gemini REST JSON: product=%s stage=%s, parts_list=%s", prod, stage, parts_list)
                    return {"product": prod, "stage": stage, "parts_list": parts_list, "raw_text": text}
                except Exception:
                    logger.exception("Failed to parse JSON from REST response; falling back to heuristic parsing")
                    try:
                        import re

                        m = re.search(r"Product\W*([^\.\n]+).*Stage\W*([^\.\n]+)", text, re.IGNORECASE | re.DOTALL)
                        if m:
                            prod = m.group(1).strip()
                            stage = m.group(2).strip()
                            return {"product": prod, "stage": stage, "parts_list": [], "raw_text": text}
                    except Exception:
                        logger.exception("Heuristic product/stage extraction failed")
                    return {"product": "", "stage": "", "parts_list": [], "raw_text": text}
        except Exception:
            pass

    # 3) Final fallback: return an empty structured result indicating we received the image
    return {"product": "", "stage": "", "parts_list": [], "raw_text": ""}


def _generate_step_by_step(image_b64: Optional[str], image_desc: Optional[str], retrieved_text: str, chat_history: List[Dict[str, str]], user_text: str, max_tokens: int = 512) -> str:
    """Produce concise, numbered assembly instructions using Gemini.

    This function synthesizes the vision analysis (`image_desc`), the retrieved
    manual text (`retrieved_text`), the conversation `chat_history` and the
    current `user_text` into a focused prompt asking for the next immediate
    action. It then asks Gemini (SDK preferred, REST fallback) to generate
    short, actionable, numbered steps.

    Returns the raw text output from Gemini (which is intended to be suitable
    for display to the user). If no generation path is available a clear stub
    prefixed with `[genai-unavailable-step]` is returned.
    """
    # build a targeted prompt for step-by-step instructions
    prompt_parts = []
    # image_desc is a dict with product/stage/raw_text
    prod = None
    stage = None
    if isinstance(image_desc, dict):
        prod = image_desc.get("product")
        stage = image_desc.get("stage")
        if prod or stage:
            prompt_parts.append(f"Image analysis: Product={prod or ''}; Stage={stage or ''}")
        elif image_desc.get("raw_text"):
            prompt_parts.append(f"Image analysis (raw): {image_desc.get('raw_text')}")
    else:
        if image_desc:
            prompt_parts.append(f"Image analysis: {image_desc}")
    if retrieved_text:
        prompt_parts.append(f"Relevant context from manuals:\n{retrieved_text}")
    if chat_history:
        hist = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in chat_history])
        prompt_parts.append(f"Chat history:\n{hist}")
    prompt_parts.append(f"User question: {user_text or 'Please provide the next assembly step.'}")

    prompt_parts.append(
        "You are an expert product assembly assistant. The user has just told you what kind of instructions they want."
        "\n\n"
        "**Analyze the User's Choice (from the last message):**"
        "\n"
        "-   If the user asks for **'detailed steps'** or **'start to finish'** or **'1'**, provide a complete, comprehensive, step-by-step assembly guide from the very beginning to the end."
        "-   If the user asks for an **'overview'** or **'brief steps'** or **'2'**, provide a high-level, bulleted list of the main assembly stages (e.g., 1. Attach legs, 2. Install shelves, 3. Mount doors)."
        "-   If the user asks for the **'first step'** or **'just the start'** or **'3'**, provide *only* the very first assembly step, but in full detail."
        "\n\n"
        "**Context to use:**"
        "\n"
        "-   Use the 'Relevant context from manuals' and your general knowledge to build the guide."
        "-   The 'Image analysis' and 'Chat history' tell you the product and parts."
        "\n\n"
        "**CRITICAL RULE: You must *only* use parts from the 'Visible Parts List' mentioned in the chat history. Do not invent any parts.**"
        "\n\n"
        "**Formatting Rules:**"
        "\n"
        "-   Use Markdown for all formatting (e.g., `**bold**`, numbered lists)."
        "-   If you cannot create a logical guide with *only* the listed parts, explain politely that the assembly cannot be completed."
    )

    final_prompt = "\n\n".join(prompt_parts)
    logger.info("Generating step-by-step with prompt length=%d", len(final_prompt))

    # Try new google.genai SDK with image inline if available
    try:
        try:
            from google import genai
            from google.genai import types
            sdk_present = True
        except Exception:
            genai = None
            types = None
            sdk_present = False

        if sdk_present:
            logger.info("Attempting step-by-step via google.genai SDK")
            try:
                client = genai.Client()
                contents = []
                # attach image part if provided
                if image_b64:
                    try:
                        image_part = types.Part.from_bytes(data=base64.b64decode(image_b64.split(",")[-1]), mime_type="image/jpeg")
                        contents.append(image_part)
                    except Exception:
                        logger.exception("Failed to build image Part from bytes; continuing without image_part")
                contents.append(final_prompt)
                resp = client.models.generate_content(model=SELECTED_MODEL, contents=contents)
                text = getattr(resp, "text", None) or str(resp)
                logger.info("Gemini SDK step-by-step response len=%d", len(text) if text else 0)
                steps = _parse_steps_from_text(text)
                return {"product": prod or "", "stage": stage or "", "steps": steps, "raw_text": text}
            except Exception:
                logger.exception("Gemini SDK step-by-step call failed; will try REST fallback")
    except Exception:
        logger.exception("Unexpected error while attempting Gemini SDK for step-by-step")

    # REST fallback for generateContent with inline image if provided
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set for REST fallback")
        import requests

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{SELECTED_MODEL}:generateContent"
        parts = []
        if image_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64.split(",")[-1]}})
        parts.append({"text": final_prompt})
        payload = {"contents": [{"parts": parts}]}
        headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
        logger.info("Calling REST generateContent for step-by-step url=%s", url)
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        logger.info("REST generateContent (step) status=%s textlen=%d", r.status_code, len(r.text) if r.text else 0)
        if r.status_code == 200:
            j = r.json()
            try:
                cand = j.get("candidates", [])
                if cand:
                    parts = cand[0].get("content", {}).get("parts", [])
                    texts = [p.get("text") for p in parts if p.get("text")]
                    text = "\n".join([t for t in texts if t])
                else:
                    text = json.dumps(j)
                steps = _parse_steps_from_text(text)
                return {"product": prod or "", "stage": stage or "", "steps": steps, "raw_text": text}
            except Exception:
                logger.exception("Failed to parse REST generateContent step response")
                return {"product": prod or "", "stage": stage or "", "steps": [], "raw_text": json.dumps(j)}
        else:
            logger.error("REST generateContent (step) failed status=%s body=%s", r.status_code, r.text[:1000])
    except Exception:
        logger.exception("REST fallback for step-by-step generation failed or not configured")

    # final stub structured response
    logger.error("All generation paths failed. Returning error to user.")
    return {"product": prod or "", "stage": stage or "", "steps": [], "raw_text": "Error with Gemini API. Please try again."}


def _generate_step_by_step_stream(image_b64: Optional[str], image_desc: Optional[str], retrieved_text: str, chat_history: List[Dict[str, str]], user_text: str):
    """Generator that streams tokens from Gemini SDK as Server-Sent Events (SSE).

    Yields SSE-formatted lines 'data: <json>\n\n' where each data is a JSON object
    with either {type: 'partial', text: '...'} or the final {type: 'done', output: {...}}.
    This only supports the SDK streaming path; REST fallback will not stream.
    """
    try:
        try:
            from google import genai
            from google.genai import types
            sdk_present = True
        except Exception:
            genai = None
            types = None
            sdk_present = False

        if not sdk_present:
            # nothing to stream
            yield f"data: {json.dumps({'type':'error','message':'Error with Gemini API. Please try again.'})}\n\n"
            return

        client = genai.Client()
        contents = []
        if image_b64:
            try:
                image_part = types.Part.from_bytes(data=base64.b64decode(image_b64.split(",")[-1]), mime_type="image/jpeg")
                contents.append(image_part)
            except Exception:
                logger.exception("Failed to build image Part for streaming; continuing without image")
        # build the same prompt as non-streaming path
        prompt_parts = []
        prod = None
        stage = None
        if isinstance(image_desc, dict):
            prod = image_desc.get("product")
            stage = image_desc.get("stage")
            if prod or stage:
                prompt_parts.append(f"Image analysis: Product={prod or ''}; Stage={stage or ''}")
            elif image_desc.get("raw_text"):
                prompt_parts.append(f"Image analysis (raw): {image_desc.get('raw_text')}")
        else:
            if image_desc:
                prompt_parts.append(f"Image analysis: {image_desc}")
        if retrieved_text:
            prompt_parts.append(f"Relevant context from manuals:\n{retrieved_text}")
        if chat_history:
            hist = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in chat_history])
            prompt_parts.append(f"Chat history:\n{hist}")
        prompt_parts.append(f"User question: {user_text or 'Please provide the next assembly step.'}")
        prompt_parts.append(
            "You are an expert product assembly assistant. The user has just told you what kind of instructions they want."
            "\n\n"
            "**Analyze the User's Choice (from the last message):**"
            "\n"
            "-   If the user asks for **'detailed steps'** or **'start to finish'** or **'1'**, provide a complete, comprehensive, step-by-step assembly guide from the very beginning to the end."
            "-   If the user asks for an **'overview'** or **'brief steps'** or **'2'**, provide a high-level, bulleted list of the main assembly stages (e.g., 1. Attach legs, 2. Install shelves, 3. Mount doors)."
            "-   If the user asks for the **'first step'** or **'just the start'** or **'3'**, provide *only* the very first assembly step, but in full detail."
            "\n\n"
            "**Context to use:**"
            "\n"
            "-   Use the 'Relevant context from manuals' and your general knowledge to build the guide."
            "-   The 'Image analysis' and 'Chat history' tell you the product and parts."
            "\n\n"
            "**CRITICAL RULE: You must *only* use parts from the 'Visible Parts List' mentioned in the chat history. Do not invent any parts.**"
            "\n\n"
            "**Formatting Rules:**"
            "\n"
            "-   Use Markdown for all formatting (e.g., `**bold**`, numbered lists)."
            "-   If you cannot create a logical guide with *only* the listed parts, explain politely that the assembly cannot be completed."
        )
        final_prompt = "\n\n".join(prompt_parts)

        # Stream from the SDK
        logger.info("Starting SDK streaming generate_content (stream=True)")
        try:
            stream = client.models.generate_content(model=SELECTED_MODEL, contents=[*contents, final_prompt], stream=True)
        except TypeError:
            # SDK might not accept stream kwarg
            logger.exception("SDK does not support stream=True")
            yield f"data: {json.dumps({'type':'error','message':'Error with Gemini API. Please try again.'})}\n\n"
            return

        full_text = ""
        try:
            for event in stream:
                # Try to extract partial text from event
                chunk_text = None
                try:
                    chunk_text = getattr(event, 'text', None)
                except Exception:
                    chunk_text = None
                if not chunk_text:
                    try:
                        # some SDKs provide delta or content parts
                        chunk_text = getattr(event, 'delta', None) or str(event)
                    except Exception:
                        chunk_text = str(event)

                if chunk_text:
                    # accumulate and yield as partial
                    full_text += chunk_text
                    payload = {'type': 'partial', 'text': chunk_text}
                    yield f"data: {json.dumps(payload)}\n\n"
        except Exception:
            logger.exception("Error while streaming from Gemini SDK")

        # After streaming finishes, parse steps and yield final event
        steps = _parse_steps_from_text(full_text)
        output = {'product': prod or '', 'stage': stage or '', 'steps': steps, 'raw_text': full_text}
        yield f"data: {json.dumps({'type':'done','output':output})}\n\n"
        return
    except Exception:
        logger.exception("Unexpected error in streaming generator")
        yield f"data: {json.dumps({'type':'error','message':'Error with Gemini API. Please try again.'})}\n\n"


def _orchestrate_assist_stream(user_text: str = "", image_b64: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 3):
    """Orchestrator wrapper that performs vision and retrieval synchronously, then returns the generator from the streaming generator."""
    chat_history = chat_history or []
    logger.info("Orchestrator (stream) starting for user_text_len=%d image_provided=%s chat_history_len=%d k=%d", len(user_text or ""), bool(image_b64), len(chat_history), k)

    image_desc = None
    if image_b64:
        image_desc = _analyze_image_with_gemini(image_b64, user_text)

    # Intent classification: only run RAG when the user is asking an assembly question
    retrieved = []
    try:
        user_intent = _classify_intent(user_text)
        logger.info("Orchestrator (stream) classified user_intent=%s", user_intent)
    except Exception:
        logger.exception("Intent classification failed; defaulting to 'assembly_question'")
        user_intent = "assembly_question"

    if user_intent == "assembly_question":
        try:
            store = SnapshotVectorStore()
            logger.info("Initializing snapshot store from %s", SNAPSHOT_DIR)
            query_parts = []
            if user_text:
                query_parts.append(user_text)
            if chat_history:
                try:
                    last_user = [m for m in chat_history if m.get('role') == 'user'][-1]
                    if last_user:
                        query_parts.append(last_user.get('content', ''))
                except Exception:
                    pass
            if image_desc:
                if isinstance(image_desc, dict):
                    prod = image_desc.get('product', '') or ''
                    stage = image_desc.get('stage', '') or ''
                    if prod or stage:
                        query_parts.append(f"Product={prod}; Stage={stage}")
                    elif image_desc.get('raw_text'):
                        query_parts.append(image_desc.get('raw_text'))
                else:
                    query_parts.append(str(image_desc))

            query = " \n ".join([p for p in query_parts if p]) or (user_text or "")
            if query.strip():
                retrieved = store.search(query, k=k)
                logger.info("Orchestrator (stream) retrieved %d documents for query", len(retrieved))
        except Exception as e:
            logger.exception("Retrieval failed in stream orchestrator: %s", e)
            retrieved = []
    else:
        logger.info("Skipping RAG search based on user intent: %s", user_intent)
        retrieved = []

    retrieved_text = "\n---\n".join([r.get('text', '') for r in retrieved])

    # Return the streaming generator
    return _generate_step_by_step_stream(image_b64, image_desc, retrieved_text, chat_history or [], user_text or "")


def _parse_steps_from_text(text: str) -> List[str]:
    """Parse numbered or bulleted steps from a text blob into a list of step strings.

    Heuristics:
      - split on lines starting with a number and period (e.g. '1. Step')
      - split on lines starting with '-' or '*' as bullets
      - fall back to splitting on double newlines if nothing else found
    """
    import re

    if not text:
        return []
    # normalize line endings
    txt = text.replace('\r\n', '\n')
    # find numbered blocks
    numbered = re.findall(r"^\s*\d+\.\s*(.+)$", txt, re.MULTILINE)
    if numbered:
        return [s.strip() for s in numbered if s.strip()]
    # find bullets
    bullets = re.findall(r"^\s*[\-\*]\s*(.+)$", txt, re.MULTILINE)
    if bullets:
        return [s.strip() for s in bullets if s.strip()]
    # fallback: split on double newlines and trim
    parts = [p.strip() for p in txt.split('\n\n') if p.strip()]
    return parts


def _classify_intent(user_text: str) -> str:
    """Lightweight intent classifier.

    Returns one of: 'assembly_question', 'context_question', 'general_chat'.

    This is intentionally small and fast (rule-based). It looks for common
    keywords to decide whether the user is asking for an assembly step (RAG
    needed) or asking about the product/context (no RAG) or just chatting.
    """
    try:
        if not user_text or not user_text.strip():
            return "general_chat"
        txt = user_text.lower()

        greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "thanks", "thank you"]
        if any(g in txt for g in greetings):
            return "general_chat"

        # assembly-oriented keywords
        assembly_keywords = [
            "step", "steps", "how to", "how do", "assemble", "assembly",
            "install", "attach", "screw", "bolt", "tighten", "next step",
            "which part", "what part", "where does", "align", "fasten"
        ]
        if any(k in txt for k in assembly_keywords):
            return "assembly_question"

        # context / product questions
        context_keywords = ["what product", "which product", "what am i working", "what is this", "product", "model", "which model"]
        if any(k in txt for k in context_keywords):
            return "context_question"

        # default to assembly_question to prefer helpfulness for short procedural queries
        return "assembly_question"
    except Exception:
        logger.exception("Intent classification error")
        return "assembly_question"


def _orchestrate_assist(user_text: str = "", image_b64: Optional[str] = None, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 3) -> Dict:
    """Orchestrate the multimodal assist flow:

    1. Optionally run vision analysis if `image_b64` is provided.
    2. Build a retrieval query from user text, last user message, and vision analysis.
    3. Query the deterministic snapshot vector store for top-k docs.
    4. Call the step-by-step generator with image, image description, retrieved text and chat history.

    Returns a dictionary ready to return from the API: {image_description, retrieved, generator_output}
    """
    chat_history = chat_history or []
    logger.info("Orchestrator: starting for user_text_len=%d image_provided=%s chat_history_len=%d k=%d", len(user_text or ""), bool(image_b64), len(chat_history), k)

    # 1) Vision analysis (if image provided)
    image_desc = None
    if image_b64:
        image_desc = _analyze_image_with_gemini(image_b64, user_text)

    # 2) Retrieval
    retrieved = []
    try:
        user_intent = _classify_intent(user_text)
        logger.info("Orchestrator classified user_intent=%s", user_intent)
    except Exception:
        logger.exception("Intent classification failed; defaulting to 'assembly_question'")
        user_intent = "assembly_question"

    if user_intent == "assembly_question":
        try:
            store = SnapshotVectorStore()
            logger.info("Initializing snapshot store from %s", SNAPSHOT_DIR)
            query_parts = []
            if user_text:
                query_parts.append(user_text)
            # include last user message from chat history if present
            if chat_history:
                try:
                    last_user = [m for m in chat_history if m.get("role") == "user"][-1]
                    if last_user:
                        query_parts.append(last_user.get("content", ""))
                except Exception:
                    pass

            # if image_desc is a dict, include product/stage; otherwise include raw text
            if image_desc:
                if isinstance(image_desc, dict):
                    prod = image_desc.get("product", "") or ""
                    stage = image_desc.get("stage", "") or ""
                    if prod or stage:
                        query_parts.append(f"Product={prod}; Stage={stage}")
                    elif image_desc.get("raw_text"):
                        query_parts.append(image_desc.get("raw_text"))
                else:
                    query_parts.append(str(image_desc))

            query = " \n ".join([p for p in query_parts if p]) or (user_text or "")
            if query.strip():
                retrieved = store.search(query, k=k)
                logger.info("Orchestrator retrieved %d documents for query", len(retrieved))
        except FileNotFoundError:
            logger.warning("Snapshot not found; continuing without retrieval")
            retrieved = []
        except Exception as e:
            logger.exception("Retrieval failed: %s", e)
            retrieved = [{"error": str(e)}]
    else:
        logger.info("Skipping RAG search based on user intent: %s", user_intent)
        retrieved = []

    # 3) Compose retrieved text for generation
    retrieved_text = "\n---\n".join([r.get("text", "") for r in retrieved])

    # 4) Generation
    gen_output = _generate_step_by_step(image_b64, image_desc, retrieved_text, chat_history or [], user_text or "")

    return {
        "image_description": image_desc,
        "retrieved": retrieved,
        "generator_output": gen_output,
    }





# Primary assistant endpoint (JSON input)


@app.post("/assist")
async def assist_json(req: AssistRequest):
    # Delegate to orchestrator. If the SDK supports streaming, return a StreamingResponse
    user_text = req.user_text or ""
    image_b64 = req.image_base64
    chat_history = req.chat_history or []
    k = int(req.k or 3)
    logger.info("/assist called user_text_len=%d image_provided=%s chat_history_len=%d k=%d", len(user_text), bool(image_b64), len(chat_history), k)

    # Try to use SDK streaming when available
    try:
        try:
            from google import genai
            sdk_present = True
        except Exception:
            sdk_present = False

        if sdk_present:
            try:
                gen = _orchestrate_assist_stream(user_text=user_text, image_b64=image_b64, chat_history=chat_history, k=k)
                return StreamingResponse(gen, media_type="text/event-stream")
            except Exception:
                logger.exception("Streaming assist failed; falling back to non-streaming response")
    except Exception:
        logger.exception("Error while detecting genai SDK for streaming")

    # Fallback: synchronous orchestrator (existing behavior)
    result = _orchestrate_assist(user_text=user_text, image_b64=image_b64, chat_history=chat_history, k=k)
    return result


@app.post("/assist-multipart")
async def assist_multipart(
    image: Optional[UploadFile] = File(None),
    user_text: Optional[str] = Form(None),
    chat_history: Optional[str] = Form(None),
    k: Optional[int] = Form(3),
):
    image_b64 = None
    if image is not None:
        data = await image.read()
        logger.info("Received multipart image size=%d bytes filename=%s", len(data), getattr(image, 'filename', None))
        image_b64 = base64.b64encode(data).decode("utf-8")

    try:
        chat_hist = json.loads(chat_history) if chat_history else []
    except Exception:
        logger.exception("Failed to parse chat_history in multipart request")
        chat_hist = []

    req = AssistRequest(user_text=user_text, image_base64=image_b64, chat_history=chat_hist, k=k)
    logger.info("/assist-multipart -> delegating to /assist JSON handler")
    return await assist_json(req)


if __name__ == "__main__":
    import uvicorn


    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
