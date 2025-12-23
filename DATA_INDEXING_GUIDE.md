# Data Indexing Guide

## How to Check Which Dataset is Currently Indexed

### Method 1: Using the Status API Endpoint

```bash
curl http://localhost:8000/status
```

Or visit: http://localhost:8000/status

**Response includes:**
```json
{
  "initialized": true,
  "data_indexed": true,
  "ollama_available": true,
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "llama3",
  "chunks_count": 79,
  "indexed_file": "CCHUB_Support_Dataset.xlsx",
  "indexed_file_path": "/path/to/data/CCHUB_Support_Dataset.xlsx",
  "indexed_at": "2025-12-02T16:30:00"
}
```

### Method 2: Check the Logs

When the server starts, check the logs in `logs/app.log`:
- Look for: `"Auto-indexing configured file: ..."`
- Look for: `"Auto-indexing completed: X chunks indexed"`
- Look for: `"Starting data indexing: ..."`

## When Data is Indexed

Data is indexed in the following scenarios:

### 1. **On Server Startup (Auto-Indexing)**
- If `AUTO_INDEX_FILE` is set in `.env`
- Only if no data is currently indexed
- Happens during `agent.initialize()`

### 2. **Via API Endpoint**
- `POST /index` - Upload a file
- `POST /index/path` - Index by file path
- **Always replaces existing data** (clears old data first)

### 3. **Manual Indexing**
- Any time you call the index endpoint, it will:
  1. Clear existing indexed data
  2. Load the new file
  3. Generate embeddings
  4. Store in ChromaDB

## Do You Need to Clear Indexed Data to Resync?

### **Yes, if you want to re-index the same file:**
- The system **automatically clears** old data when you index a new file
- Each indexing operation replaces all existing data
- You don't need to manually clear before indexing

### **When to Clear Manually:**

1. **To force re-indexing on startup:**
   ```bash
   # Delete ChromaDB data
   rm -rf chroma_db/
   
   # Restart server (will auto-index if configured)
   python main.py
   ```

2. **If auto-indexing is being skipped:**
   - Auto-indexing only runs if `has_data() == False`
   - If data exists, it skips auto-indexing
   - Clear the data to force re-indexing

### **How to Clear Indexed Data:**

**Option 1: Delete ChromaDB directory**
```bash
rm -rf chroma_db/
```

**Option 2: Re-index via API (automatically clears)**
```bash
curl -X POST "http://localhost:8000/index/path" \
  -F "file_path=./data/CCHUB_Support_Dataset.xlsx"
```

**Option 3: Index a different file (replaces automatically)**
```bash
curl -X POST "http://localhost:8000/index/path" \
  -F "file_path=./data/your_new_file.csv"
```

## Understanding the Indexing Process

1. **File Loading**: CSV/Excel file is loaded into pandas DataFrame
2. **Preprocessing**: Data is cleaned and normalized
3. **Chunking**: Each row becomes a searchable text chunk
4. **Embedding**: Text chunks are converted to vector embeddings
5. **Storage**: Embeddings stored in ChromaDB with metadata
6. **Metadata**: Source file name, path, and timestamp are stored

## Troubleshooting

### "Data already indexed" message
- This means ChromaDB has existing data
- Auto-indexing is skipped to preserve existing data
- To re-index: Clear `chroma_db/` or use `/index` endpoint

### File not auto-indexing on startup
- Check `.env` file has `AUTO_INDEX_FILE` set
- Check file path is correct (relative to project root)
- Check logs for errors
- Verify file exists at the specified path

### Want to switch datasets
- Simply index the new file via API - it automatically replaces old data
- Or delete `chroma_db/` and restart server


