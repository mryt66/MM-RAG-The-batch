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

Set Gemini API key:
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

Run the application:
```powershell
streamlit run app.py
```

The app will open at http://localhost:8501

Run the evaluation dashboard:
```powershell
streamlit run evaluation/evaluation_dashboard.py --server.port 8502
```

The dashboard will open at http://localhost:8502

## Configuration

- Text Embeddings: Qwen/Qwen3-Embedding-0.6B
- Image Embeddings: clip-ViT-B-32 (CLIP)
- LLM: gemini-2.5-flash (Google Gemini)
- Vector Database: ChromaDB with local persistent storage

## How It Works

1. Data Ingestion: Scrapes 320 articles from The Batch
2. Embedding Generation: Creates vector representations (text: Qwen3-Embedding-0.6B, images: CLIP)
3. Vector Storage: Indexes in ChromaDB for fast similarity search
4. Query Processing: User query retrieves top-3 relevant articles
5. Context Construction: Builds prompt with retrieved articles
6. Answer Generation: Gemini generates contextual answer
7. Automatic Logging: Saves to database for evaluation
8. Metric Computation: Evaluates quality on next app startup
