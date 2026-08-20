import websocket
import json
import time
from websocket import rel

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get("type") == "accident_detected":
            receipt_time = time.time()
            detection_time = float(data["timestamp"])  # This is a string in ISO format? Actually, it's a string like "2026-08-14T19:24:02.407621"
            # We need to convert the ISO string to a timestamp
            from datetime import datetime
            dt = datetime.fromisoformat(data["timestamp"])
            detection_timestamp = dt.timestamp()
            latency_ms = (receipt_time - detection_timestamp) * 1000
            print(f"Latency: {latency_ms:.2f} ms")
    except Exception as e:
        print(f"Error processing message: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_reason):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8000/ws/alerts",
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.run_forever()