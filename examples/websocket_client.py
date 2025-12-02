"""
Example WebSocket client for Core Query Bot.

Demonstrates how to connect and interact with the WebSocket endpoint.

Author: Blessing Ajala - Software Engineer
GitHub: https://github.com/Oyelamin
LinkedIn: https://www.linkedin.com/in/blessphp/
Twitter: @Blessin06147308
"""

import asyncio
import json
import uuid

import websockets


async def chat_with_bot(session_id: str = None):
    """
    Connect to WebSocket and have a conversation with the bot.
    
    Args:
        session_id: Optional session ID. If None, a new one will be generated.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        print(f"Generated session ID: {session_id}")
    
    # Connect to WebSocket with session_id in query parameter
    uri = f"ws://localhost:8000/ws?session_id={session_id}"
    
    async with websockets.connect(uri) as websocket:
        print(f"✅ Connected to WebSocket (Session: {session_id})")
        
        # Wait for connection confirmation
        response = await websocket.recv()
        print(f"📨 Server: {response}\n")
        
        # Interactive chat loop
        while True:
            try:
                # Get user input
                user_query = input("You: ").strip()
                
                if not user_query:
                    continue
                
                if user_query.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break
                
                if user_query.lower() == "clear":
                    # Clear conversation history
                    message = {
                        "type": "clear_history",
                        "session_id": session_id
                    }
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    print(f"📨 Server: {response}\n")
                    continue
                
                # Send query
                message = {
                    "type": "query",
                    "query": user_query,
                    "session_id": session_id,
                    "include_metrics": False
                }
                
                await websocket.send(json.dumps(message))
                
                # Receive response
                response = await websocket.recv()
                data = json.loads(response)
                
                if data.get("type") == "error":
                    print(f"❌ Error: {data.get('error')}\n")
                elif data.get("type") == "response":
                    print(f"🤖 Bot: {data.get('response')}\n")
                else:
                    print(f"📨 Server: {response}\n")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    import sys
    
    # Get session_id from command line if provided
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    print("🚀 Core Query Bot - WebSocket Client")
    print("=" * 50)
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'exit' or 'quit' to disconnect")
    print("=" * 50)
    print()
    
    try:
        asyncio.run(chat_with_bot(session_id))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

