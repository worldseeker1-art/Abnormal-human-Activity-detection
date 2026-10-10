"""Start the surveillance agent and operator console."""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
