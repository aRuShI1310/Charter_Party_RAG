# Charter Party RAG - System Architecture

## Project Flow Diagram

```mermaid
graph TD
    A["📄 Input Documents<br/>(PDF, DOCX, TXT)"] 
    B["🔄 Extract Text<br/>(OCR)"]
    C["✂️ Chunk Text<br/>(Split into passages)"]
    D["🔗 Generate Embeddings<br/>(Convert to vectors)"]
    E["💾 Vector Store<br/>(Neon PostgreSQL)"]
    F["❓ User Question"]
    G["🔍 Vector Search<br/>(Find similar chunks)"]
    H["📚 Retrieve Top-K<br/>Relevant Passages"]
    I["🤖 LLM Generation<br/>(GPT-4o-mini)"]
    J["✅ Final Answer<br/>(with citations)"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    F --> G
    E --> G
    G --> H
    H --> I
    I --> J
    
    style A fill:#e1f5ff
    style E fill:#fff3e0
    style J fill:#e8f5e9
    style I fill:#f3e5f5
```

## Simplified Flow

**3 Main Steps:**

1. **INGEST** → Documents → Extract → Chunk → Embed → Store in Database
2. **RETRIEVE** → Question → Search vectors → Get top passages
3. **ANSWER** → Context + Question → LLM → Final Answer

## Component Overview

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Input** | Charter party documents | PDF, DOCX, TXT |
| **Text Extraction** | OCR & parsing | Google Cloud API |
| **Chunking** | Break into passages | LangChain |
| **Embedding** | Convert to vectors | OpenAI API |
| **Vector Store** | Semantic search database | Neon + pgvector |
| **Retrieval** | Find relevant chunks | Vector similarity |
| **LLM** | Generate answers | GPT-4o-mini |

---

**Key Benefit:** RAG allows the LLM to answer questions using your specific charter party documents, ensuring accurate, cited responses.
