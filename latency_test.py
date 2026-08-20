import asyncio
import json
import websockets
import time

async def listen():
    uri = "ws://localhost:8001/ws/alerts"
    latencies = []
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to WebSocket")
            start = time.time()
            while time.time() - start < 30:  # run for 30 seconds
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(message)
                    if data.get("type") == "accident_detected":
                        ts_str = data.get("timestamp")
                        if ts_str:
                            # parse ISO string to struct_time
                            # Using time.strptime for simplicity; assumes format YYYY-MM-DDTHH:MM:SS.ffffff
                            try:
                                # strip Z if present
                                if ts_str.endswith('Z'):
                                    ts_str = ts_str[:-1] + '+00:00'
                                # parse using datetime
                                from datetime import datetime
                                dt = datetime.fromisoformat(ts_str)
                                # Convert to seconds since epoch
                                detection_sec = dt.timestamp()
                                receipt_sec = time.time()
                                latency_ms = (receipt_sec - detection_sec) * 1000.0
                                latencies.append(latency_ms)
                                print(f"Detection: confidence={data.get('confidence'):.3f}, latency={latency_ms:.1f} ms")
                            except Exception as e:
                                print(f"Timestamp parse error: {e}")
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    break
    except Exception as e:
        print(f"WebSocket error: {e}")
        return
    if latencies:
        import statistics
        avg = statistics.mean(latencies)
        mx = max(latencies)
        p95 = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 2 else mx
        print("\n=== Latency Statistics (30s) ===")
        print(f"Samples: {len(latencies)}")
        print(f"Average: {avg:.1f} ms")
        print(f"Max: {mx:.1f} ms")
        print(f"95th percentile: {p95:.1f} ms")
        under_2s = all(l < 2000 for l in latencies)
        print(f"All under 2000 ms? {under_2s}")
    else:
        print("No detections captured.")

if __name__ == "__main__":
    asyncio.run(listen())