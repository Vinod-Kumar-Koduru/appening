# Agentic AI RAG API

A document question-answering API built with FastAPI, LangGraph, OpenAI embeddings, and Pinecone. It retrieves PDF excerpts, drafts an answer from those excerpts, and asks a separate grading step whether the answer is supported. If the first answer is not supported, it retries with more retrieved chunks; if the second answer is still unsupported, it returns a refusal.

## Request flow

1. `src/ingestion.py` reads the PDF, splits it into 800-character chunks with 100-character overlap, and stores the chunks with their page metadata in Pinecone.
2. `src/graph.py` retrieves the three closest chunks and drafts an answer using only that context.
3. A grounding check scores the answer. An unsupported answer triggers one retrieval retry with six chunks; a second unsupported answer is refused.
4. `app.py` returns the query, answer, context chunks, and confidence score from `POST /chat`.

The confidence score is an LLM judge's estimate of evidence support, not a calibrated probability or a guarantee that an answer is correct.

## Run locally

Use Python 3.10 or newer. Create and activate a virtual environment, then install the packages:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, use `python -m venv .venv` and `source .venv/bin/activate` instead.

Copy `.env` to `.env` and set `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME`. Keep `.env` private; it must not be committed to GitHub.

Download the [Agentic AI eBook PDF](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf), place it at `data/Ebook-Agentic-AI.pdf`, then run ingestion from the project root:

```bash
python -m src.ingestion
```

The Pinecone index must use the embedding dimension for `text-embedding-3-small` (1536). Start the API:

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to try `POST /chat`. `GET /health` reports whether the graph initialized.

## Example response

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "...",
  "retrieved_context_chunks": ["[Page 3] ..."],
  "confidence_score": 0.91
}
```

## Sample queations

What is the core definition of Agentic AI as outlined in the eBook?What are the main architectural components required to build agentic systems?
