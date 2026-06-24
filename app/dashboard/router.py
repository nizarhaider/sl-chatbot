from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from app.dashboard.state import dashboard_state

router = APIRouter()


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@router.get("/dashboard/calls")
def live_calls() -> dict:
    return dashboard_state.snapshot()


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Call Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f7f8;
      color: #1d252d;
    }
    body {
      margin: 0;
      min-height: 100vh;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      background: #ffffff;
      border-bottom: 1px solid #dce3e8;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      padding: 20px 24px 32px;
    }
    .status {
      color: #52606d;
      font-size: 14px;
      white-space: nowrap;
    }
    .calls {
      display: grid;
      gap: 12px;
    }
    .call {
      background: #ffffff;
      border: 1px solid #dce3e8;
      border-radius: 8px;
      overflow: hidden;
    }
    .callHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 16px;
      border-bottom: 1px solid #e7ecef;
    }
    .callTitle {
      display: grid;
      gap: 2px;
      min-width: 0;
    }
    .phone {
      font-size: 15px;
      font-weight: 650;
      overflow-wrap: anywhere;
    }
    .callId {
      color: #697783;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .pill {
      border-radius: 999px;
      background: #e8f5ee;
      color: #17623a;
      font-size: 12px;
      font-weight: 650;
      padding: 4px 9px;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .transcript {
      display: grid;
      gap: 10px;
      padding: 14px 16px 16px;
    }
    .event {
      display: grid;
      grid-template-columns: 96px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }
    .speaker {
      color: #52606d;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .text {
      font-size: 15px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .empty {
      color: #697783;
      padding: 48px 0;
      text-align: center;
    }
    @media (max-width: 640px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      main {
        padding: 14px;
      }
      .callHeader {
        align-items: flex-start;
        flex-direction: column;
      }
      .event {
        grid-template-columns: 1fr;
        gap: 3px;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Live Calls</h1>
    <div class="status" id="status">Connecting...</div>
  </header>
  <main>
    <div class="calls" id="calls"></div>
  </main>
  <script>
    const callsEl = document.getElementById("calls");
    const statusEl = document.getElementById("status");

    function formatTime(seconds) {
      return new Date(seconds * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
      });
    }

    function render(data) {
      const calls = data.calls || [];
      statusEl.textContent = `${calls.length} live call${calls.length === 1 ? "" : "s"} | Updated ${new Date().toLocaleTimeString()}`;
      if (!calls.length) {
        callsEl.innerHTML = '<div class="empty">No live calls</div>';
        return;
      }
      callsEl.innerHTML = calls.map(call => `
        <section class="call">
          <div class="callHeader">
            <div class="callTitle">
              <div class="phone">${escapeHtml(call.caller_phone || "Unknown caller")}</div>
              <div class="callId">${escapeHtml(call.call_id)} | Started ${formatTime(call.started_at)}</div>
            </div>
            <div class="pill">${escapeHtml(call.status)}</div>
          </div>
          <div class="transcript">
            ${(call.transcript || []).length ? call.transcript.map(event => `
              <div class="event">
                <div class="speaker">${escapeHtml(event.speaker)}<br>${formatTime(event.timestamp)}</div>
                <div class="text">${escapeHtml(event.text)}</div>
              </div>
            `).join("") : '<div class="empty">Waiting for transcript</div>'}
          </div>
        </section>
      `).join("");
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function refresh() {
      try {
        const response = await fetch("/dashboard/calls", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        render(await response.json());
      } catch (error) {
        statusEl.textContent = `Dashboard disconnected | ${error.message}`;
      }
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""
