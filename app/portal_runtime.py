import os
import re
import subprocess
import sys
import threading
import time

import httpx
import psutil
import uvicorn


base = os.environ["PORTAL_URL"].rstrip("/")
auth = {"Authorization": f"Bearer {os.environ['PORTAL_RUNTIME_TOKEN']}"}
with httpx.Client(timeout=30) as client:
    response = client.get(f"{base}/api/runtime/config", headers=auth)
    response.raise_for_status()
    config = response.json()
for key, value in config["env"].items():
    if value:
        os.environ[key] = str(value)
os.environ["PORTAL_MAX_CALLS"] = str(config["max_calls"])

from app.api.app import create_app

app = create_app()
runtime_url = ""
tunnels = []


def tunnel():
    global runtime_url
    process = subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:8081"], stderr=subprocess.PIPE, text=True)
    tunnels.append(process)
    for line in process.stderr:
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if match:
            runtime_url = match[0]
    runtime_url = ""


def heartbeat():
    process = psutil.Process()
    with httpx.Client(timeout=20) as client:
        while True:
            try:
                from app.voice.agent import voice_agent
                status = "error" if getattr(app.state, "voice_startup_error", "") else "ready" if getattr(app.state, "voice_ready", False) else "warming_up"
                result = client.post(f"{base}/api/runtime/heartbeat", headers=auth, json={
                    "version": config["version"], "active_calls": len(voice_agent.active_calls),
                    "cpu_percent": psutil.cpu_percent(), "memory_mb": round(process.memory_info().rss / 1048576, 1),
                    "status": status, "runtime_url": runtime_url,
                    "error": "Gemini startup failed. Check the configured API key and model." if status == "error" else "",
                })
                result.raise_for_status()
                control = result.json()
                os.environ["PORTAL_MAX_CALLS"] = str(control.get("max_calls", config["max_calls"]))
                if control.get("restart"):
                    for child in tunnels:
                        child.terminate()
                    os.execv(sys.executable, [sys.executable, "-m", "app.portal_runtime"])
            except Exception as exc:
                print(f"Portal heartbeat unavailable: {type(exc).__name__}", flush=True)
            time.sleep(20)


if os.environ.get("PORTAL_DISABLE_NAMED_TUNNEL") != "1" and os.environ.get("CLOUDFLARED_TUNNEL_TOKEN"):
    tunnels.append(subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "run", "--token", os.environ["CLOUDFLARED_TUNNEL_TOKEN"]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
threading.Thread(target=tunnel, daemon=True).start()
threading.Thread(target=heartbeat, daemon=True).start()
uvicorn.run(app, host="0.0.0.0", port=8081)
