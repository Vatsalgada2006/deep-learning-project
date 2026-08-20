import websocket
import json
import time

def on_message(ws, message):
    print(f"Received message: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_reason):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")
    # Optionally send a message
    # ws.send(json.dumps({"type": "ping"}))

if __name__ == "__main__":
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp("ws://localhost:8000/ws/alerts",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever()