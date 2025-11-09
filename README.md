# 🤖 Product Assembly Assistant

A multimodal, RAG-powered chatbot designed to guide you through complex product assembly, one step at a time. Upload a photo of your product, and the assistant will identify what you're building, what stage you're at, and give you the exact next step.

## ✨ Core Features

* **Multimodal Input:** Accepts both text queries ("What's next?") and image uploads of your product.
* **Smart Image Analysis:** Uses the Gemini model to analyze uploaded images, identify the product (e.g., "Wooden cabinet"), and determine the current assembly stage (e.g., "Assembling doors and shelves").
* **RAG-Powered Guidance:** Your questions are answered using a **Retrieval-Augmented Generation (RAG)** system. The assistant searches a vector database of your *own* PDF manuals to find the most relevant instructions.
* **Intelligent & Context-Aware:**
    * **Chat History:** Remembers the full conversation, so you can ask follow-up questions.
    * **Intent Classification:** It doesn't *always* run RAG. If you just say "hi" or ask "what product am I working on?", it uses its context-awareness to answer, saving the RAG search for when you're actually asking for assembly help.
* **Modern, Responsive UI:**
    * Light & Dark mode theme toggle.
    * Drag-and-drop file uploads.
    * Streaming responses that feel like a modern chatbot.
    * Full Markdown rendering for clear, formatted instructions.

---

## 🚀 How it Works

This application provides a "smart" layer over your product manuals.


1.  **Upload & Ask:** The user uploads an image and/or asks a question (e.g., "I'm stuck here, what's next?").
2.  **Image Analysis:** The FastAPI backend sends the image to Gemini for analysis. It returns a structured JSON object like `{"product": "storage cabinet", "stage": "attaching the cabinet door"}`.
3.  **Intent Classification:** The backend classifies the user's text to decide if it's a `general_chat`, a `context_question`, or an `assembly_question`.
4.  **Vector Search (RAG):** If it's an `assembly_question`, the assistant queries its vector database (the `RAG_store` folder). It searches for text chunks from your manuals that are semantically similar to your question *and* the image analysis.
5.  **Generate Response:** A final, comprehensive prompt is built. It includes:
    * The image analysis ("You are working on a cabinet...")
    * The RAG context ("...and the manual says to use screw 'B'...")
    * The chat history
    * The user's question
6.  **Stream to UI:** This complete context is sent to Gemini, which generates a detailed, step-by-step answer. This answer is streamed token-by-token back to the React frontend.

---

## 🛠️ Getting Started

### 1. Prerequisites

* Python 3.8+
* Node.js (v18+)
* An active **Google API Key** with the Gemini API enabled.

### 2. Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone https://your-repo-url/product-assembly-assistant.git
    cd product-assembly-assistant
    ```

2.  **Install Python dependencies:**
    *(It's highly recommended to use a virtual environment)*
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt`, create one and add: `fastapi`, `uvicorn`, `python-dotenv`, `google-generativeai`, `sentence-transformers`, `numpy`, `pypdf`, `requests`)*

3.  **Set up your Environment:**
    Create a file named `.env` in the project's root directory and add your API key:
    ```
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

### 3. Frontend Setup

1.  **Navigate to the frontend directory:**
    *(Assuming your React app is in a `frontend/` sub-folder)*
    ```bash
    cd frontend
    ```

2.  **Install Node modules:**
    ```bash
    npm install
    ```

---

## 🗂️ Building Your Vector Database (Ingestion)

Before you can run the assistant, you **must** create the vector database from your product manuals.

This project uses a "snapshot" vector store, which means we run the `ingest.py` script *once* to process all your manuals and create local files. The backend then reads these files for all RAG queries.

1.  **Add Your Manuals:**
    Place all your product assembly PDF files inside the `manuals/` directory.

2.  **Run the Ingestion Script:**
    From the project's root directory, run:
    ```bash
    python ingest.py
    ```

3.  **What it Does:**
    This script will:
    * Find all `.pdf` files in the `manuals/` folder.
    * Load and parse the text from each file.
    * Split the text into smaller, overlapping chunks.
    * Use the `all-MiniLM-L6-v2` model to create vector embeddings for every chunk.
    * Save everything into the `RAG_store/` folder as a set of files:
        * `embeddings.npy` (The vector data)
        * `texts.json` (The raw text chunks)
        * `metadatas.json` (Info about where the chunks came from)
        * `ids.json` (Unique IDs for each chunk)

Your backend is now ready to perform RAG searches using this local data.

---

## 🏃 Running the Application

You'll need two terminals open.

1.  **Terminal 1: Start the Backend (FastAPI)**
    From the project root:
    ```bash
    python main.py
    ```
    Your backend is now running at `http://localhost:8000`.

2.  **Terminal 2: Start the Frontend (React)**
    From the `frontend/` directory:
    ```bash
    npm run dev
    ```
    Your app is now accessible in your browser at `http://localhost:5173` (or whatever URL Vite provides).

---

## 🥞 Tech Stack

* **Backend:** FastAPI, Python, Uvicorn
* **Frontend:** React (with Vite), ReactMarkdown
* **LLM & Vision:** Google Gemini
* **Vector Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Vector Store:** Custom NumPy / JSON snapshot store