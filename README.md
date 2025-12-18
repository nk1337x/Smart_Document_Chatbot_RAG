# 📚 RAG Document Chatbot

A ChatGPT-like interface for chatting with your documents using **free and open-source** tools. Built with Python and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **📤 File Upload**: Upload PDF and TXT files
- **🔍 Text Extraction**: Automatically extract text from documents
- **✂️ Smart Chunking**: Split text into manageable chunks with overlap
- **🧠 Embeddings**: Generate embeddings using HuggingFace Sentence Transformers (free!)
- **💾 Vector Database**: Store and search embeddings with FAISS (local, no API needed)
- **🤖 LLM Generation**: Generate answers using free models (Flan-T5 or Ollama)
- **💬 ChatGPT-like UI**: Beautiful chat interface with conversation history
- **📦 Caching**: Cached embeddings to avoid recomputation
- **📑 Source Citations**: View source chunks for each answer

## 🛠️ Technology Stack (All Free & Open Source)

| Component | Technology | License |
|-----------|------------|---------|
| Web UI | Streamlit | Apache 2.0 |
| Embeddings | Sentence Transformers | Apache 2.0 |
| Vector Store | FAISS | MIT |
| LLM | Flan-T5 / Ollama | Apache 2.0 / MIT |
| PDF Processing | PyPDF2, pdfplumber | BSD / MIT |

## 📋 Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB recommended for better performance)
- ~2GB disk space for models (downloaded on first run)

## 🚀 Installation

### 1. Clone or Download the Project

```bash
cd "AI-Powered Document Chatbot-RAG"
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first installation may take a few minutes as it downloads PyTorch and other dependencies.

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### Step 1: Upload Documents
1. Click on the file uploader in the sidebar
2. Select one or more PDF or TXT files
3. Files are processed locally - your data never leaves your machine!

### Step 2: Configure Settings
- **Chunk Size**: Adjust how text is split (500-1000 chars recommended)
- **Chunk Overlap**: Overlap between chunks for context continuity
- **Embedding Model**: Choose based on speed vs quality needs
- **Language Model**: Select Flan-T5 (works everywhere) or Ollama (if installed)

### Step 3: Process Documents
Click "Process Documents" to:
1. Extract text from your files
2. Split into chunks
3. Generate embeddings
4. Build the search index

### Step 4: Start Chatting!
Ask questions about your documents in the chat input. The system will:
1. Find relevant chunks using semantic search
2. Generate an answer based on the retrieved context
3. Show source citations (optional)

## ⚙️ Configuration Options

### Embedding Models

| Model | Speed | Quality | Dimensions |
|-------|-------|---------|------------|
| all-MiniLM-L6-v2 | ⚡ Fast | Good | 384 |
| all-mpnet-base-v2 | 🐢 Slower | Better | 768 |
| paraphrase-MiniLM-L6-v2 | ⚡ Fast | Good (paraphrases) | 384 |

### Language Models

#### HuggingFace (Flan-T5) - Default
- **Pros**: Works out of the box, no setup required
- **Cons**: Smaller model, simpler responses
- **Best for**: Quick testing, limited hardware

#### Ollama (Optional)
- **Pros**: Larger models (Llama 2, Mistral), better responses
- **Cons**: Requires separate installation
- **Best for**: Production use, better quality answers

**To use Ollama:**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama2`
3. Start Ollama server
4. Select "Ollama" in the app settings

## 📁 Project Structure

```
AI-Powered Document Chatbot-RAG/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── cache/                # Cached embeddings (created automatically)
└── utils/
    ├── __init__.py
    ├── document_processor.py  # PDF/TXT text extraction
    ├── text_chunker.py        # Text splitting with overlap
    ├── embedding_manager.py   # HuggingFace embeddings
    ├── vector_store.py        # FAISS vector database
    └── llm_handler.py         # LLM response generation
```

## 🔧 Troubleshooting

### "Model download taking too long"
- First run downloads models (~500MB for embeddings, ~1GB for LLM)
- This is a one-time download; subsequent runs use cached models

### "Out of memory error"
- Try a smaller chunk size (300-400)
- Use `all-MiniLM-L6-v2` embedding model
- Close other applications

### "FAISS not found"
```bash
pip install faiss-cpu
```

### "PyTorch installation issues"
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### "Ollama connection failed"
- Ensure Ollama is installed and running
- Check if Ollama server is active: `ollama list`
- Fall back to HuggingFace if issues persist

## 🎨 Customization

### Change the System Prompt
Edit `utils/llm_handler.py` and modify `SYSTEM_PROMPT` and `RAG_PROMPT_TEMPLATE`.

### Add New File Types
Extend `utils/document_processor.py` to support additional formats (DOCX, HTML, etc.).

### Use Different Vector Database
Replace FAISS with Chroma or other vector stores in `utils/vector_store.py`.

## 📊 Performance Tips

1. **For faster processing**: Use `all-MiniLM-L6-v2` embeddings
2. **For better answers**: Use larger chunk sizes (800-1000)
3. **For large documents**: Process in batches
4. **Enable caching**: Already enabled by default to avoid reprocessing

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

This project is licensed under the MIT License - free to use, modify, and distribute.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Amazing Python web framework
- [HuggingFace](https://huggingface.co/) - Free ML models and transformers
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [Ollama](https://ollama.ai/) - Easy local LLM deployment

---

**Made with ❤️ using free and open-source tools**
