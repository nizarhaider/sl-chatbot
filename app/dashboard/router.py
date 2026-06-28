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
  <title>SerendibAI Voice Monitor</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f6f7;
      color: #18222c;
      --bg: #f4f6f7;
      --panel: #ffffff;
      --panel-soft: #f8faf9;
      --border: #dce5e8;
      --border-strong: #c9d6db;
      --text-muted: #5f6f7a;
      --text-soft: #7a8892;
      --green: #157347;
      --green-soft: #e7f5ed;
      --blue: #1f6feb;
      --blue-soft: #eaf2ff;
      --amber: #a65f00;
      --amber-soft: #fff4df;
      --red: #b42318;
      --red-soft: #fff0ee;
      --shadow: 0 10px 30px rgba(22, 34, 44, 0.08);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(180deg, #eef3f2 0, var(--bg) 320px),
        var(--bg);
    }
    header {
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .headerInner {
      max-width: 1280px;
      margin: 0 auto;
      padding: 18px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }
    .mark {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #173d34;
      color: #ffffff;
      font-weight: 800;
      letter-spacing: 0;
      flex: 0 0 auto;
    }
    .brandText {
      min-width: 0;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .subtitle {
      margin-top: 2px;
      color: var(--text-muted);
      font-size: 13px;
    }
    main {
      max-width: 1280px;
      margin: 0 auto;
      padding: 22px 24px 34px;
    }
    .status {
      align-items: center;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      color: var(--text-muted);
      display: inline-flex;
      font-size: 13px;
      gap: 8px;
      padding: 8px 11px;
      white-space: nowrap;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 0 4px var(--green-soft);
    }
    .dot.offline {
      background: var(--red);
      box-shadow: 0 0 0 4px var(--red-soft);
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }
    .stat {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 15px 16px;
      min-width: 0;
    }
    .statLabel {
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0;
      text-transform: uppercase;
    }
    .statValue {
      margin-top: 8px;
      font-size: 28px;
      font-weight: 760;
      line-height: 1;
    }
    .statMeta {
      margin-top: 7px;
      color: var(--text-soft);
      font-size: 12px;
      min-height: 16px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .toolbar {
      align-items: center;
      display: flex;
      gap: 12px;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .sectionTitle {
      font-size: 14px;
      font-weight: 750;
      letter-spacing: 0;
    }
    .filter {
      display: inline-flex;
      background: #e9eff1;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 3px;
    }
    .filter button {
      appearance: none;
      background: transparent;
      border: 0;
      border-radius: 6px;
      color: var(--text-muted);
      cursor: pointer;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      min-width: 70px;
      padding: 7px 10px;
    }
    .filter button.active {
      background: var(--panel);
      color: #18222c;
      box-shadow: 0 1px 3px rgba(22, 34, 44, 0.12);
    }
    .calls {
      display: grid;
      gap: 14px;
    }
    .call {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .callHeader {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 15px 16px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(180deg, #ffffff, var(--panel-soft));
    }
    .callTitle {
      display: grid;
      gap: 5px;
      min-width: 0;
    }
    .phone {
      align-items: center;
      display: flex;
      gap: 8px;
      font-size: 16px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }
    .phone::before {
      content: "";
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--text-soft);
      flex: 0 0 auto;
    }
    .call.active .phone::before {
      background: var(--green);
      box-shadow: 0 0 0 4px var(--green-soft);
    }
    .callId {
      color: var(--text-soft);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .metaRow {
      align-items: center;
      color: var(--text-muted);
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      font-size: 12px;
      margin-top: 2px;
    }
    .pill {
      border-radius: 999px;
      background: var(--green-soft);
      color: var(--green);
      font-size: 12px;
      font-weight: 760;
      padding: 6px 10px;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .call.ended .pill {
      background: #edf1f3;
      color: #53636e;
    }
    .call.connecting .pill {
      background: var(--amber-soft);
      color: var(--amber);
    }
    .callBody {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 190px;
      gap: 0;
    }
    .transcript {
      display: grid;
      gap: 12px;
      padding: 16px;
    }
    .event {
      display: flex;
      gap: 10px;
      max-width: 920px;
    }
    .event.assistant {
      margin-left: 34px;
    }
    .event.caller {
      margin-right: 34px;
    }
    .avatar {
      align-items: center;
      border-radius: 8px;
      display: flex;
      flex: 0 0 34px;
      height: 34px;
      justify-content: center;
      width: 34px;
      font-size: 12px;
      font-weight: 800;
      background: var(--blue-soft);
      color: var(--blue);
      margin-top: 2px;
    }
    .event.assistant .avatar {
      background: var(--green-soft);
      color: var(--green);
    }
    .bubble {
      min-width: 0;
      background: #f4f8fb;
      border: 1px solid #dce8ef;
      border-radius: 8px;
      padding: 10px 12px;
    }
    .event.assistant .bubble {
      background: #f4faf6;
      border-color: #d8eadf;
    }
    .speaker {
      align-items: center;
      color: var(--text-muted);
      display: flex;
      font-size: 12px;
      font-weight: 700;
      gap: 8px;
      margin-bottom: 5px;
      text-transform: uppercase;
    }
    .text {
      font-size: 15px;
      line-height: 1.48;
      overflow-wrap: anywhere;
    }
    .side {
      border-left: 1px solid var(--border);
      background: #fbfcfc;
      padding: 16px;
    }
    .sideGrid {
      display: grid;
      gap: 12px;
    }
    .sideItem {
      min-width: 0;
    }
    .sideLabel {
      color: var(--text-soft);
      font-size: 11px;
      font-weight: 760;
      text-transform: uppercase;
    }
    .sideValue {
      color: #27333d;
      font-size: 13px;
      font-weight: 700;
      margin-top: 3px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .empty {
      color: var(--text-soft);
      padding: 48px 16px;
      text-align: center;
      background: var(--panel);
      border: 1px dashed var(--border-strong);
      border-radius: 8px;
    }
    .transcript .empty {
      border: 0;
      background: transparent;
      padding: 28px 0;
    }
    .hidden {
      display: none;
    }
    @media (max-width: 860px) {
      .headerInner {
        align-items: flex-start;
        flex-direction: column;
        padding: 16px;
      }
      main {
        padding: 16px;
      }
      .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .callBody {
        grid-template-columns: 1fr;
      }
      .side {
        border-left: 0;
        border-top: 1px solid var(--border);
      }
    }
    @media (max-width: 560px) {
      main {
        padding: 12px;
      }
      .stats {
        grid-template-columns: 1fr;
      }
      .toolbar {
        align-items: stretch;
        flex-direction: column;
      }
      .callHeader {
        align-items: flex-start;
        flex-direction: column;
      }
      .event {
        margin-left: 0 !important;
        margin-right: 0 !important;
      }
      .filter {
        width: 100%;
      }
      .filter button {
        flex: 1;
        min-width: 0;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="headerInner">
      <div class="brand">
        <div class="mark">H</div>
        <div class="brandText">
          <h1>SerendibAI Voice Monitor</h1>
          <div class="subtitle">WhatsApp call sessions and live transcripts</div>
        </div>
      </div>
      <div class="status" id="status"><span class="dot" id="statusDot"></span><span id="statusText">Connecting...</span></div>
    </div>
  </header>
  <main>
    <section class="stats" aria-label="Call metrics">
      <div class="stat">
        <div class="statLabel">Active</div>
        <div class="statValue" id="activeCount">0</div>
        <div class="statMeta" id="activeMeta"></div>
      </div>
      <div class="stat">
        <div class="statLabel">Total</div>
        <div class="statValue" id="totalCount">0</div>
        <div class="statMeta" id="totalMeta"></div>
      </div>
      <div class="stat">
        <div class="statLabel">Messages</div>
        <div class="statValue" id="messageCount">0</div>
        <div class="statMeta" id="messageMeta"></div>
      </div>
      <div class="stat">
        <div class="statLabel">Latest</div>
        <div class="statValue" id="latestTime">--</div>
        <div class="statMeta" id="latestMeta"></div>
      </div>
    </section>
    <div class="toolbar">
      <div class="sectionTitle">Sessions</div>
      <div class="filter" role="group" aria-label="Filter sessions">
        <button class="active" data-filter="all" type="button">All</button>
        <button data-filter="active" type="button">Active</button>
        <button data-filter="ended" type="button">Ended</button>
      </div>
    </div>
    <div class="calls" id="calls"></div>
  </main>
  <script>
    const callsEl = document.getElementById("calls");
    const statusTextEl = document.getElementById("statusText");
    const statusDotEl = document.getElementById("statusDot");
    const activeCountEl = document.getElementById("activeCount");
    const activeMetaEl = document.getElementById("activeMeta");
    const totalCountEl = document.getElementById("totalCount");
    const totalMetaEl = document.getElementById("totalMeta");
    const messageCountEl = document.getElementById("messageCount");
    const messageMetaEl = document.getElementById("messageMeta");
    const latestTimeEl = document.getElementById("latestTime");
    const latestMetaEl = document.getElementById("latestMeta");
    const filterButtons = Array.from(document.querySelectorAll(".filter button"));
    let currentFilter = "all";
    let lastData = { calls: [] };

    filterButtons.forEach(button => {
      button.addEventListener("click", () => {
        currentFilter = button.dataset.filter;
        filterButtons.forEach(item => item.classList.toggle("active", item === button));
        render(lastData);
      });
    });

    function formatTime(seconds) {
      if (!seconds) return "--";
      return new Date(seconds * 1000).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
      });
    }

    function formatDuration(startedAt, endedAt) {
      if (!startedAt) return "--";
      const end = endedAt || Date.now() / 1000;
      const total = Math.max(0, Math.round(end - startedAt));
      const minutes = Math.floor(total / 60);
      const seconds = total % 60;
      return `${minutes}:${String(seconds).padStart(2, "0")}`;
    }

    function shortId(callId) {
      if (!callId) return "unknown";
      return callId.length > 24 ? `${callId.slice(0, 18)}...${callId.slice(-6)}` : callId;
    }

    function speakerInitial(speaker) {
      return String(speaker || "?").slice(0, 1).toUpperCase();
    }

    function updateStats(calls) {
      const active = calls.filter(call => call.status === "active");
      const ended = calls.filter(call => call.status === "ended");
      const messages = calls.reduce((sum, call) => sum + (call.transcript || []).length, 0);
      const latest = calls[0];

      activeCountEl.textContent = active.length;
      activeMetaEl.textContent = active.length ? `${active[0].caller_phone || "Unknown caller"} on call` : "No live calls";
      totalCountEl.textContent = calls.length;
      totalMetaEl.textContent = `${ended.length} ended`;
      messageCountEl.textContent = messages;
      messageMetaEl.textContent = calls.length ? "Transcript events" : "";
      latestTimeEl.textContent = latest ? formatTime(latest.updated_at || latest.started_at) : "--";
      latestMetaEl.textContent = latest ? (latest.caller_phone || shortId(latest.call_id)) : "No activity";
    }

    function render(data) {
      lastData = data;
      const calls = data.calls || [];
      const active = calls.filter(call => call.status === "active").length;
      updateStats(calls);
      statusDotEl.classList.remove("offline");
      statusTextEl.textContent = `${active} active | Updated ${new Date().toLocaleTimeString()}`;
      const visibleCalls = calls.filter(call => currentFilter === "all" || call.status === currentFilter);
      if (!visibleCalls.length) {
        callsEl.innerHTML = '<div class="empty">No matching call sessions</div>';
        return;
      }
      callsEl.innerHTML = visibleCalls.map(call => {
        const transcript = call.transcript || [];
        const callerMessages = transcript.filter(event => event.speaker === "caller").length;
        const assistantMessages = transcript.filter(event => event.speaker === "assistant").length;
        const latestEvent = transcript[transcript.length - 1];
        return `
        <section class="call ${escapeHtml(call.status || "unknown")}">
          <div class="callHeader">
            <div class="callTitle">
              <div class="phone">${escapeHtml(call.caller_phone || "Unknown caller")}</div>
              <div class="callId">${escapeHtml(shortId(call.call_id))}</div>
              <div class="metaRow">
                <span>Started ${formatTime(call.started_at)}</span>
                ${call.ended_at ? `<span>Ended ${formatTime(call.ended_at)}</span>` : ""}
                <span>Duration ${formatDuration(call.started_at, call.ended_at)}</span>
              </div>
            </div>
            <div class="pill">${escapeHtml(call.status)}</div>
          </div>
          <div class="callBody">
            <div class="transcript">
              ${transcript.length ? transcript.map(event => `
                <div class="event ${escapeHtml(event.speaker)}">
                  <div class="avatar">${escapeHtml(speakerInitial(event.speaker))}</div>
                  <div class="bubble">
                    <div class="speaker"><span>${escapeHtml(event.speaker)}</span><span>${formatTime(event.timestamp)}</span></div>
                    <div class="text">${escapeHtml(event.text)}</div>
                  </div>
                </div>
              `).join("") : '<div class="empty">Waiting for transcript</div>'}
            </div>
            <aside class="side">
              <div class="sideGrid">
                <div class="sideItem">
                  <div class="sideLabel">Caller Turns</div>
                  <div class="sideValue">${callerMessages}</div>
                </div>
                <div class="sideItem">
                  <div class="sideLabel">Assistant Turns</div>
                  <div class="sideValue">${assistantMessages}</div>
                </div>
                <div class="sideItem">
                  <div class="sideLabel">Last Update</div>
                  <div class="sideValue">${formatTime(call.updated_at || call.started_at)}</div>
                </div>
                <div class="sideItem">
                  <div class="sideLabel">Latest Text</div>
                  <div class="sideValue" title="${escapeHtml(latestEvent ? latestEvent.text : "")}">${escapeHtml(latestEvent ? latestEvent.text : "No transcript yet")}</div>
                </div>
              </div>
            </aside>
          </div>
        </section>
      `;
      }).join("");
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
        statusDotEl.classList.add("offline");
        statusTextEl.textContent = `Disconnected | ${error.message}`;
      }
    }

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""
