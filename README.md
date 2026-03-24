# Code-RAG

**Local code assistant powered by Retrieval-Augmented Generation.**

Code-RAG indexes your codebase into a vector database and uses retrieval-augmented generation to answer questions about your code — all running locally.

## How It Works

1. **Index** — Your code is chunked, embedded, and stored in a local vector database
2. **Retrieve** — When you ask a question, the most relevant code chunks are fetched via similarity search
3. **Generate** — Retrieved context is fed to the LLM alongside your query for grounded, accurate responses

## Configuration

All parameters are tunable via config/config.yaml:

| Parameter | What it does |
|-----------|-------------|
| MODEL_NAME | LLM model for code generation |
| TOP_K | Number of retrieved chunks to include as context |
| TOP_P | Nucleus sampling threshold |
| MAX_LENGTH | Maximum response token length |
| TEMPERATURE | Creativity vs. precision control |

## Project Structure

```
code-RAG/
 src/              # Core RAG pipeline
 config/           # Configuration files
 models/           # Model artifacts
 data/             # Source data for indexing
 vectordb/         # Vector database storage
 tests/            # Test suite
 .github/workflows # CI/CD
```

## Setup

```bash
git clone https://github.com/brettleehari/code-RAG.git
cd code-RAG
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python src/main.py
```

## Why I Built This

RAG is the most common pattern in production AI applications today. I wanted hands-on experience with the full pipeline — chunking strategies, embedding models, vector storage, retrieval tuning, and grounded generation. This project taught me the tradeoffs that matter when building RAG products: chunk size vs context quality, retrieval precision vs recall, and the cost of re-indexing.

## Author

**Hariprasad Sudharshan** - [GitHub](https://github.com/brettleehari)

## License

MIT
