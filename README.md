# Core Query Bot - AI Agent for CSV/Excel Data

A minimal RAG-based AI agent that allows you to query structured CSV/Excel data using natural language. Built with FastAPI, semantic search, and local LLM (Ollama).

## Features

- 🔍 **Semantic Search**: Uses embeddings for intelligent data retrieval
- 🤖 **Local LLM**: Powered by Ollama (completely free)
- 💬 **Conversational AI**: Maintains conversation history for natural follow-up questions
- 🔌 **WebSocket Support**: Real-time bidirectional communication with session management
- 📊 **Performance Tracking**: Monitor query performance and metrics
- 🚫 **Smart Filtering**: Automatically detects when information is not available
- 🚀 **FastAPI API**: RESTful API with automatic documentation
- 💾 **Persistent Storage**: ChromaDB for vector storage
- 🎯 **Session Management**: Track conversations per device/session ID

## Architecture

- **RAG (Retrieval-Augmented Generation)**: Search your data, then generate answers
- **Modular Components**: Clean separation of concerns
- **Performance Monitoring**: Track response times and similarity scores

## Prerequisites

1. **Python 3.8+**
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Install a model: `ollama pull llama3` (or `mistral`)

## Installation

1. **Clone and navigate to the project:**
   ```bash
   cd corequery-bot
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Ollama (if not running):**
   ```bash
   ollama serve
   ```

5. **Pull a model (if not done already):**
   ```bash
   ollama pull llama3
   # or
   ollama pull mistral
   ```

## Usage

### Start the API Server

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### 1. Check Status
```bash
GET /status
```

#### 2. Index a CSV/Excel File

**Option A: Upload file**
```bash
POST /index
Content-Type: multipart/form-data
Body: file=<your_file.csv>
```

**Option B: Provide file path**
```bash
POST /index/path
Content-Type: application/x-www-form-urlencoded
Body: file_path=/path/to/your/file.csv
```

#### 3. Query the Data

**POST request:**
```bash
POST /query
Content-Type: application/json
Body: {
  "query": "What products are available?",
  "include_metrics": true
}
```

**GET request (simple):**
```bash
GET /query/simple?q=What products are available?&include_metrics=true
```

### Example Usage

#### Using curl:

```bash
# 1. Index a file
curl -X POST "http://localhost:8000/index/path" \
  -F "file_path=./data/products.csv"

# 2. Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What products cost under $50?", "include_metrics": true}'
```

#### Using Python:

```python
import requests

# Index file
response = requests.post(
    "http://localhost:8000/index/path",
    data={"file_path": "./data/products.csv"}
)
print(response.json())

# Query (with session_id for conversation history)
session_id = "my-device-123"  # Use same ID for follow-up conversations

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What products are available?",
        "include_metrics": True,
        "session_id": session_id
    }
)
print(response.json())

# Follow-up query (AI remembers previous conversation)
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "Which ones are under $50?",  # Follow-up question
        "include_metrics": False,
        "session_id": session_id  # Same session_id
    }
)
print(response.json())
```

#### Using the Interactive Docs:

Visit http://localhost:8000/docs for an interactive API documentation where you can test all endpoints directly.

### Conversational Queries with Session ID

The API supports conversational queries by maintaining conversation history per session. Use the same `session_id` across multiple queries to enable follow-up conversations:

```python
import requests

# First query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "How do I create an account?",
        "session_id": "my-device-123",  # Unique session ID
        "include_metrics": False
    }
)
print(response.json()["response"])

# Follow-up query (AI remembers previous conversation)
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What if I forget my password?",  # Follow-up question
        "session_id": "my-device-123",  # Same session ID
        "include_metrics": False
    }
)
print(response.json()["response"])  # AI will reference previous conversation
```

#### Using WebSocket (Real-time Conversational):

For real-time conversational queries with context awareness, use the WebSocket endpoint:

```python
import asyncio
import json
import websockets

async def chat():
    session_id = "your-device-id-or-uuid"  # Unique identifier for your device/session
    uri = f"ws://localhost:8000/ws?session_id={session_id}"
    
    async with websockets.connect(uri) as websocket:
        # Wait for connection confirmation
        response = await websocket.recv()
        print(response)
        
        # Send a query
        message = {
            "type": "query",
            "query": "How do I create an account?",
            "session_id": session_id,
            "include_metrics": False
        }
        await websocket.send(json.dumps(message))
        
        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Bot: {data['response']}")
        
        # Send follow-up query (will have access to previous conversation)
        message = {
            "type": "query",
            "query": "What if I forget my password?",
            "session_id": session_id
        }
        await websocket.send(json.dumps(message))
        
        response = await websocket.recv()
        data = json.loads(response)
        print(f"Bot: {data['response']}")

asyncio.run(chat())
```

**WebSocket Message Types:**

- `query`: Send a query to the bot
  ```json
  {
    "type": "query",
    "query": "Your question here",
    "session_id": "your-session-id",
    "include_metrics": false
  }
  ```

- `clear_history`: Clear conversation history for the session
  ```json
  {
    "type": "clear_history",
    "session_id": "your-session-id"
  }
  ```

- `ping`: Check connection (server responds with `pong`)
  ```json
  {
    "type": "ping",
    "session_id": "your-session-id"
  }
  ```

**Example Client:**

See `examples/websocket_client.py` for a complete interactive WebSocket client example.

Run it with:
```bash
python examples/websocket_client.py [session_id]
```

### Conversation History & Follow-ups

The bot maintains conversation history per `session_id`, enabling natural follow-up conversations:

**Features:**
- ✅ Maintains last 10 messages per session
- ✅ Natural reference to previous conversations
- ✅ Context-aware responses
- ✅ Session-based conversation tracking
- ✅ Auto-cleanup of inactive sessions (1 hour timeout)

**Example Conversation Flow:**

```
User: "How do I create an account?"
Bot: "Great question! To create an account, you can click 'Sign Up' on the homepage..."

User: "What if I forget my password?"
Bot: "Sure! If you forget your password, click 'Forgot Password?' on the login page..."

User: "How long does that take?"
Bot: "The password reset process is instant! You'll receive a reset link right away..."
```

The bot will naturally reference previous messages and build on the conversation context.

## Configuration

Copy `.env.example` to `.env` and customize settings (optional):

```bash
cp .env.example .env
```

Then edit `.env` to customize settings:

```env
# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Similarity threshold (0.0 - 1.0)
SIMILARITY_THRESHOLD=0.7

# Number of top matches to retrieve
TOP_K_MATCHES=5

# LLM model (Ollama)
LLM_MODEL=llama3

# Ollama base URL
OLLAMA_BASE_URL=http://localhost:11434

# Auto-index file on startup (optional)
AUTO_INDEX_FILE=data/sample_faq.csv

# Logging
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_TRACKING=true
```

### Auto-Indexing on Startup

Set `AUTO_INDEX_FILE` in your `.env` file to automatically index a CSV/Excel file when the app starts:

```env
AUTO_INDEX_FILE=data/sample_faq.csv
```

This will:
- Automatically load and index the file on startup
- Skip if data is already indexed
- Log errors if the file is not found

## Project Structure

```
corequery-bot/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration
│   │
│   ├── api/                   # API layer
│   │   ├── __init__.py
│   │   ├── app.py            # FastAPI application
│   │   └── schemas.py        # Request/response models
│   │
│   ├── core/                  # Core business logic
│   │   ├── __init__.py
│   │   ├── agent.py          # Main orchestrator
│   │   ├── query_processor.py # Semantic search
│   │   └── session_manager.py # Conversation history management
│   │
│   ├── data/                  # Data processing
│   │   ├── __init__.py
│   │   ├── loader.py         # CSV/Excel loading
│   │   └── embeddings.py     # Embedding generation
│   │
│   ├── llm/                   # LLM integration
│   │   ├── __init__.py
│   │   └── client.py         # Ollama client
│   │
│   └── utils/                 # Utilities
│       ├── __init__.py
│       ├── exceptions.py    # Custom exceptions
│       ├── logger.py         # Logging setup
│       └── performance.py    # Performance tracking
│
├── data/                      # Your CSV/Excel files
├── chroma_db/                 # Vector database (auto-created)
├── logs/                      # Log files (auto-created)
├── metrics/                   # Performance metrics (auto-created)
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## How It Works

1. **Indexing**: 
   - Load CSV/Excel file
   - Generate embeddings for each row
   - Store in ChromaDB vector database

2. **Querying**:
   - User asks a question
   - Generate query embedding
   - Search for similar content in vector DB
   - Check similarity threshold (0.7)
   - If found: Generate LLM response
   - If not found: Return "Unable to process" message

3. **Performance Tracking**:
   - Track response times
   - Log similarity scores
   - Export metrics to JSON files

## Response Format

```json
{
  "query": "What products are available?",
  "response": "Based on the data, we have Product A, Product B...",
  "status": "success",
  "similarity_score": 0.85,
  "matches_found": 3,
  "meets_threshold": true,
  "metrics": {
    "response_time_ms": 1200,
    "embedding_time_ms": 50,
    "search_time_ms": 100,
    "llm_time_ms": 1050,
    "similarity_score": 0.85,
    "status": "success"
  }
}
```

## Troubleshooting

### Ollama not running
```bash
# Start Ollama
ollama serve
```

### Model not found
```bash
# Pull the model
ollama pull llama3
```

### Port already in use
```bash
# Change port in main.py or use:
uvicorn main:app --port 8001
```

### Low similarity scores
- Adjust `SIMILARITY_THRESHOLD` in config (lower = more lenient)
- Ensure your CSV has relevant data
- Check if query matches data format

## Performance

- **Embedding Model**: `all-MiniLM-L6-v2` (~80MB, fast)
- **Vector DB**: ChromaDB (persistent, local)
- **LLM**: Ollama (local, free)
- **Typical Response Time**: 1-3 seconds

## Contributing

🤝 **This project is open for collaboration!** 

We welcome contributions, suggestions, and improvements. Whether you want to:
- Add new features
- Fix bugs
- Improve documentation
- Optimize performance
- Suggest enhancements

Feel free to:
- Open an issue to discuss ideas
- Submit a pull request
- Share feedback and suggestions

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Author & Contact

**Blessing Ajala - Software Engineer**

- 👨‍💻 **GitHub**: [@Oyelamin](https://github.com/Oyelamin)
- 💼 **LinkedIn**: [blessphp](https://www.linkedin.com/in/blessphp/)
- 🐦 **Twitter**: [@Blessin06147308](https://x.com/Blessin06147308)

Feel free to reach out for questions, collaborations, or just to say hello! 😊

## License

MIT

## Support

For issues or questions, please check the logs in the `logs/` directory or open an issue on GitHub.

