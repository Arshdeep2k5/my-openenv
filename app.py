# app.py — Re-export from server.app for local development compatibility
# The actual FastAPI app is in server/app.py for Docker deployment

from server.app import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
