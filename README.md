# Charter_Party_RAG

Intelligent Legal Document Question-Answering System for Charter Parties using Retrieval-Augmented Generation (RAG).

This repository implements a RAG-based system tuned for charter party contracts and related legal documents. It provides tools to ingest documents, store embeddings, and answer natural-language questions using an LLM augmented by relevant retrieved passages.

## Features

- Ingest and index charter party documents (PDF, DOCX, TXT).
- Store embeddings in a vector store (Neon/Postgres-compatible).
- Query using an LLM with retrieval-augmented generation.
- Configurable to use LangSmith for observability and Google APIs for document OCR/transforms.
- Simple CLI and/or programmatic API to run ingestion and QA pipelines.

## Prerequisites

- Python 3.8+ (or the language/runtime your project uses)
- pip (or poetry)
- PostgreSQL-compatible database (Neon is used in examples)
- API keys:
  - LANGSMITH_API_KEY — for LangSmith observability (optional but recommended)
  - GOOGLE_API_KEY — for Google Cloud APIs (optional — only if using Google OCR / Document AI)
- Network access to your database and any external APIs

## Environment variables

Create a `.env` file in the project root (or set env vars in your deployment environment):

```env
# .env (example)
LANGSMITH_API_KEY=your_langsmith_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
NEON_DB_CONNECTION_STRING=postgresql://{user}:{pwd}@{domain}/neondb?sslmode=require
# Optional additional settings
VECTOR_STORE_TABLE=embeddings
EMBEDDING_MODEL=openai-embedding-model-or-other
LLM_MODEL=gpt-4o-mini  # or other LLM identifier
```

- NEON_DB_CONNECTION_STRING: the PostgreSQL connection string for your vector store / metadata store. Replace `{user}`, `{pwd}`, `{domain}` accordingly.
- LANGSMITH_API_KEY: for logging and observability of LLM calls (if using LangSmith).
- GOOGLE_API_KEY: for Google Cloud services (OCR, Document AI) if used.

## Installation

1. Clone the repo:
   git clone https://github.com/aRuShI1310/Charter_Party_RAG.git
   cd Charter_Party_RAG

2. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows

3. Install dependencies:
   - If you have a `requirements.txt`:
     pip install -r requirements.txt
   - Or if you use Poetry:
     poetry install

If your project has a different language/tooling, follow the repository-specific install instructions (package.json, pyproject.toml, etc.).

## Quick start

1. Ensure `.env` is configured with the values above.
2. Prepare documents to ingest (put PDFs/TXT/DOCX in a folder, e.g. `./data/docs`).
3. Run the ingestion script:

   Example (Python, adjust to your project):
   python scripts/ingest.py --input ./data/docs --db "$NEON_DB_CONNECTION_STRING"

   The ingestion step should:
   - Extract text (OCR if needed)
   - Chunk text into passages
   - Create embeddings and store them in the vector table

4. Start a query session / run the QA script:

   python scripts/query.py --question "What is the laytime clause in Charter Party X?" --db "$NEON_DB_CONNECTION_STRING"

   That script should:
   - Retrieve top-k relevant passages via the vector store
   - Call the configured LLM to produce a final, cited answer

Adjust the CLI flags to match the actual scripts in your repository.

## Example usage (programmatic)

Pseudocode illustrating the flow:

1. Load environment and initialize db/vector store client.
2. Ingest documents:
   - extract_text(file)
   - chunks = chunk_text(text)
   - embeddings = embed(chunks)
   - upsert_to_vector_store(embeddings, metadata)
3. Query:
   - hits = vector_store.search(query_embedding, top_k=5)
   - answer = llm.generate_with_context(hits, prompt_template)

## Data model & storage

- Vector store: stores embeddings, chunk text, and metadata (document id, page, offset).
- Metadata store (same DB or separate): stores document-level metadata and audit logs.
- Optional observability: LangSmith traces for prompts and LLM responses.

## Testing

- Add unit tests for ingestion, chunking, and retrieval logic.
- Add integration tests that use a test database (or a local Postgres/Neon instance).
- Use mock/stub for LLM and embedding APIs in CI.

## Deployment

- Containerize the service with Docker (recommended).
- Ensure secrets (API keys, DB credentials) are injected via environment variables or secret manager.
- For production scale:
  - Use an external vector database if needed (e.g., Pinecone, Milvus) or scale your Postgres.
  - Add caching and request rate limiting.

## Troubleshooting

- Connection errors to NEON_DB: verify connection string, network rules, and credentials.
- Missing embeddings: confirm the embedder API key and model are configured properly.
- Poor QA results: adjust chunk size, increase retrieval top_k, or prompt templates.

## Contributing

Contributions are welcome! Suggested workflow:
1. Fork the repository
2. Create a feature branch: git checkout -b feat/my-change
3. Add tests for changes where applicable
4. Open a pull request describing your changes

Please follow the repository's code style and commit conventions.



## Acknowledgements

- Retrieval-Augmented Generation (RAG) best practices
- LangSmith for observability
- Any libraries or toolkits you used (list them in detail in the final README)

## Contact

Maintainer: aRuShI1310 (GitHub)  
For questions or help: open an issue in this repository.
