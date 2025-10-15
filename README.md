# Multimodal RAG System for The Batch

An end-to-end **Retrieval-Augmented Generation (RAG)** system with multimodal capabilities (text + images) built for The Batch news articles. Features a chat interface powered by **Gemini 2.5 Flash** with automatic evaluation and analytics.

## 🎯 Features

- **Multimodal Search**: Text-only, image-only, or combined text+image queries
- **Smart Weighting**: Automatic optimization of text/image importance based on similarity
- **Chat Interface**: Conversation memory with context-aware responses
- **Automatic Evaluation**: All interactions logged and evaluated automatically
- **Analytics Dashboard**: Comprehensive metrics (Context Relevance, Groundedness, Answer Relevance, Token Usage)
- **LLM Integration**: Gemini 2.5 Flash for answer generation

## 📁 Project Structure

```
multimodal_rag/
├── app.py                      # Main Streamlit chat application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── rag/                        # RAG core modules
│   ├── scrape_min.py          # Web scraper for The Batch articles
│   ├── compute_embeddings.py  # Generate text & image embeddings
│   └── retrieval.py           # Retrieval and prompt construction
├── evaluation/                 # Evaluation system
│   ├── database_logger.py     # SQLite logging & evaluation
│   ├── evaluation_dashboard.py # Analytics dashboard
│   └── eval_questions.txt     # Test questions for evaluation
├── data/                       # Article data
│   ├── the_batch_articles_min.json
│   ├── metadata.json
│   └── image_metadata.json
├── embeddings/                 # Precomputed embeddings
│   ├── embeddings.npz         # Text embeddings (Sentence-BERT)
│   └── image_embeddings.npz   # Image embeddings (CLIP)
├── chroma_db/                  # ChromaDB vector database
└── rag_logs.db                 # SQLite logs of all interactions
```

## 🚀 Quick Start

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

The app will open at **http://localhost:8501**

### 4. First-Time Setup (in the UI)

1. Click **"Compute Text Embeddings"** - Generates embeddings for all articles (~2-3 minutes)
2. Click **"Compute Image Embeddings"** - Generates embeddings for images (~2-3 minutes)
3. Click **"Load Index"** - Loads the vector database
4. Start chatting!

## 💬 Using the Chat Interface

### Text-Only Query
Type your question:
```
What are the latest developments in AI?
```

### Image-Only Query
1. Upload an image
2. Type a space (" ") and press Enter

### Multimodal Query (Text + Image)
1. Upload an image
2. Type your question
3. System automatically balances text/image weights

### Follow-Up Questions
The system remembers your last 3 exchanges:
```
You: "What is machine learning?"
Bot: [Explains ML]

You: "What are the applications?"  ← Remembers you're asking about ML
Bot: [Lists ML applications]
```

## 📊 Evaluation System

### Automatic Evaluation
- **Every interaction is logged** to `rag_logs.db`
- **Auto-evaluated on app startup** (runs once per session)
- Evaluates unevaluated interactions automatically

### Evaluation Metrics
1. **Context Relevance (0-1)**: How relevant are retrieved contexts?
2. **Groundedness (0-1)**: Is the answer supported by contexts?
3. **Answer Relevance (0-1)**: Does the answer address the question?
4. **Token Usage**: Input/output/total tokens for cost tracking

### View Analytics Dashboard
```powershell
streamlit run evaluation/evaluation_dashboard.py --server.port 8502
```
Dashboard opens at **http://localhost:8502** with:
- Leaderboard table with all metrics
- Performance charts and analytics
- Detailed per-interaction breakdown
- "Evaluate All" button for manual evaluation

## 🛠️ Advanced Usage

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

## 🔧 Configuration

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

## 📝 Requirements

- **Python**: 3.10+
- **API Key**: Google Gemini API key (free tier available)
- **Memory**: 4GB+ RAM recommended
- **Storage**: ~2GB for embeddings and database
- **GPU**: Optional (CUDA-compatible for faster embeddings)

## 🔍 Troubleshooting

### App won't start
- Check Python version: `python --version` (should be 3.10+)
- Verify dependencies: `pip install -r requirements.txt`
- Check API key is set: `echo $env:GEMINI_API_KEY`

### No embeddings found
- Click "Compute Text Embeddings" in the UI
- Or run: `python rag/compute_embeddings.py`

### Evaluation not working
- Check if `rag_logs.db` exists (created automatically)
- View logs with: `sqlite3 rag_logs.db "SELECT * FROM interactions;"`

### Out of memory
- Reduce batch size in `rag/compute_embeddings.py`
- Close other applications
- Use CPU instead of GPU

## 📚 Components

### Main Application (`app.py`)
- Streamlit chat interface
- Multimodal query handling
- Conversation history management
- Automatic interaction logging
- Auto-evaluation on startup

### RAG Module (`rag/`)
- **scrape_min.py**: Scrapes articles from The Batch
- **compute_embeddings.py**: Generates embeddings with BERT & CLIP
- **retrieval.py**: Semantic search and prompt construction

### Evaluation Module (`evaluation/`)
- **database_logger.py**: SQLite logging and metric computation
- **evaluation_dashboard.py**: Interactive analytics dashboard
- **eval_questions.txt**: Test questions for evaluation

## 🎓 How It Works

1. **Data Ingestion**: Scrapes 320 articles from The Batch
2. **Embedding Generation**: Creates vector representations (text: BERT, images: CLIP)
3. **Vector Storage**: Indexes in ChromaDB for fast similarity search
4. **Query Processing**: User query → retrieve top-3 relevant articles
5. **Context Construction**: Builds prompt with retrieved articles
6. **Answer Generation**: Gemini generates contextual answer
7. **Automatic Logging**: Saves to database for evaluation
8. **Metric Computation**: Evaluates quality on next app startup

## ✅ Evaluation Results

Sample metrics from testing:
- **Context Relevance**: 0.85 avg (excellent retrieval)
- **Groundedness**: 0.72 avg (good factual accuracy)
- **Answer Relevance**: 0.88 avg (highly relevant responses)
- **Avg Tokens**: ~900 per interaction

## 🌟 Key Highlights

- ✅ Complete end-to-end RAG pipeline
- ✅ Multimodal capabilities (text + images)
- ✅ Smart automatic weighting algorithm
- ✅ Conversation memory (last 3 exchanges)
- ✅ Automatic evaluation with 4 metrics
- ✅ Professional UI with Streamlit
- ✅ Comprehensive analytics dashboard
- ✅ Production-ready logging system

## 📄 License

This project is for educational purposes. The Batch articles are property of DeepLearning.AI.

## 🤝 Support

For issues or questions:
1. Check this README
2. View evaluation dashboard for system health
3. Check `rag_logs.db` for interaction logs
