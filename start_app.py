#!/usr/bin/env python3
import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=False
    )
