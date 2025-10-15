# Multimodal RAG System for The Batch

An end-to-end Retrieval-Augmented Generation (RAG) system with multimodal capabilities (text + images) built for The Batch news articles. Features a chat interface powered by Gemini 2.5 Flash with automatic evaluation and analytics.

## Features

- Multimodal Search: Text-only, image-only, or combined text+image queries
- Smart Weighting: Automatic optimization of text/image importance based on similarity
- Chat Interface: Conversation memory with context-aware responses
- Automatic Evaluation: All interactions logged and evaluated automatically
- Analytics Dashboard: Comprehensive metrics (Context Relevance, Groundedness, Answer Relevance, Token Usage)
- LLM Integration: Gemini 2.5 Flash for answer generation

## Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2. Set Up API Key

```powershell
# Set Gemini API key (required)
$env:GEMINI_API_KEY="your-api-key-here"
```

### 3. Run the Application

```powershell
streamlit run app.py
```

The app will open at http://localhost:8501

## Advanced Usage

### Scrape Fresh Articles
```powershell
python rag/scrape_min.py --start 1 --end 320 --out data/the_batch_articles_min.json --delay 0.5
```

### Rebuild Embeddings
```powershell
python rag/compute_embeddings.py
```

### Run Standalone Evaluation
```powershell
# Evaluate test questions
python evaluation/evaluate.py

# View results in terminal
python evaluation/evaluate.py --view
```

## Configuration

### Models Used
- **Text Embeddings**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Image Embeddings**: `clip-ViT-B-32` (CLIP)
- **LLM**: `gemini-2.5-flash` (Google Gemini)

### Vector Database
- **Technology**: ChromaDB
- **Storage**: Local persistent storage in `chroma_db/`
- **Collections**: Separate collections for text and images

### GPU Support
- Automatically detects and uses GPU if available
- Falls back to CPU (slower but works)
- Adjustable batch sizes for different hardware

## Requirements

- Python 3.10+
- Google Gemini API key (free tier available)
- 4GB+ RAM recommended
- ~2GB storage for embeddings and database
- GPU optional (CUDA-compatible for faster embeddings)

## How It Works

1. Data Ingestion: Scrapes 320 articles from The Batch
2. Embedding Generation: Creates vector representations (text: BERT, images: CLIP)
3. Vector Storage: Indexes in ChromaDB for fast similarity search
4. Query Processing: User query retrieves top-3 relevant articles
5. Context Construction: Builds prompt with retrieved articles
6. Answer Generation: Gemini generates contextual answer
7. Automatic Logging: Saves to database for evaluation
8. Metric Computation: Evaluates quality on next app startup
