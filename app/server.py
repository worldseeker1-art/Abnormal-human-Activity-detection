"""Operator console HTTP API."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import SurveillanceAgent

STATIC = Path(__file__).resolve().parent / "static"
DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard"
agent = SurveillanceAgent(loiter_seconds=8.0)
agent.start()

app = FastAPI(title="Agents / Repos")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD / "index.html")


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD / "index.html")


@app.get("/atm")
def atm() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "running": agent.status()["running"], "camera": agent.camera.mode}


@app.get("/api/status")
def status() -> dict:
    return agent.status()


@app.get("/api/snapshot.jpg")
def snapshot() -> Response:
    data = agent.snapshot()
    if not data:
        return Response(status_code=503)
    return Response(content=data, media_type="image/jpeg")


@app.get("/api/stream")
def stream() -> StreamingResponse:
    def frames():
        while True:
            data = agent.snapshot()
            if data:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            time.sleep(0.08)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")
