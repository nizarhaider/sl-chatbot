import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import aiohttp
from aiortc import RTCSessionDescription
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
from pipecat.transports.whatsapp.client import WhatsAppClient

load_dotenv()

from app.agent import prewarm, run_agent
from app.config import CASE_CALL_SECONDS, DEFAULT_MAX_CALLS, ICE_SERVERS
from app.store import call_state
from app.tools import CallContext, connection, whatsapp_api

logger = logging.getLogger(__name__)


async def query(sql, params=()):
    def execute():
        with connection() as db:
            result = db.execute(sql, params)
            return result.fetchone() if result.description else None

    return await asyncio.to_thread(execute)


@dataclass
class Call:
    context: CallContext
    connection: SmallWebRTCConnection
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    timer: asyncio.TimerHandle | None = None
    outbound: bool = False
    answered: bool = False
    failed: bool = False


class Calls:
    def __init__(self):
        self.connections = {}
        self.reservations = set()
        self.tasks = set()
        self.lock = asyncio.Lock()
        self.client = None

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.finished)
        return task

    def finished(self, task):
        self.tasks.discard(task)
        if not task.cancelled() and task.exception():
            logger.error("Call task failed: %s", type(task.exception()).__name__)

    async def reserve(self, key):
        async with self.lock:
            if (
                key in self.connections
                or key in self.reservations
                or key in call_state.calls
            ):
                return False
            if len(self.connections) + len(self.reservations) >= int(
                os.environ.get("MAX_CALLS", str(DEFAULT_MAX_CALLS))
            ):
                raise HTTPException(503, "Voice agents are busy")
            self.reservations.add(key)
            return True

    def register(self, context, peer, outbound=False):
        call = Call(context, peer, outbound=outbound)
        self.connections[context.call_id] = call
        call_state.start_call(context)
        if not outbound:
            call.ready.set()
        call.task = self.spawn(self.run(call))
        if outbound:
            call.timer = asyncio.get_running_loop().call_later(
                CASE_CALL_SECONDS,
                lambda: self.spawn(self.close_call(context.call_id, terminate=True)),
            )
        return call

    async def run(self, call):
        try:
            await run_agent(call.connection, call.context, call.ready)
        except Exception:
            call.failed = True
            logger.exception("Voice pipeline failed")
            call_state.emit(
                call.context.call_id,
                "pipeline.error",
                {"error": "Voice pipeline failed"},
            )
        finally:
            await self.close_call(call.context.call_id, terminate=True)

    async def close_call(self, call_id, terminate=False):
        call = self.connections.pop(call_id, None)
        if not call:
            return
        if call.timer:
            call.timer.cancel()
        if terminate:
            await whatsapp_api.send_call_action(call_id, "terminate")
        try:
            await call.connection.disconnect()
        except Exception:
            logger.exception("Call transport cleanup failed")
        if call.task and call.task is not asyncio.current_task():
            call.task.cancel()
            await asyncio.gather(call.task, return_exceptions=True)
        call_state.end_call(call_id)
        if call.outbound:
            await query(
                """update portal_case_requests set state=%s,error=%s where call_id=%s
                   and runtime_agent_id=%s and direction='outbound' and state='calling'""",
                (
                    "completed" if call.answered and not call.failed else "failed",
                    ""
                    if call.answered
                    else "The call was not answered. Please try again.",
                    call_id,
                    os.environ["VOICE_AGENT_ID"],
                ),
            )

    async def case_request(self, request_id):
        return await query(
            """select q.*,a.id as agent_id,a.customer_id from portal_case_requests q
               join portal_agent_cases c on c.slug=q.case_slug and c.runtime_agent_id=q.runtime_agent_id
               join portal_agents a on a.id=c.agent_id where q.id=%s and q.runtime_agent_id=%s
               and q.case_slug='sales' and q.direction='outbound' and q.expires_at>now()
               and a.status<>'archived'""",
            (request_id, os.environ["VOICE_AGENT_ID"]),
        )

    async def fail_case(self, request_id, message):
        await query(
            "update portal_case_requests set state='failed',error=%s where id=%s and runtime_agent_id=%s",
            (message, request_id, os.environ["VOICE_AGENT_ID"]),
        )

    async def request_case(self, request_id):
        row = await self.case_request(request_id)
        if not row or row["state"] not in ("preparing", "permission"):
            return
        reserved = False
        try:
            permission = await whatsapp_api.call_permissions(row["phone"])
            allowed = {
                a["action_name"]
                for a in permission.get("actions", [])
                if a.get("can_perform_action")
            }
            if "start_call" not in allowed:
                if row["state"] == "preparing":
                    if "send_call_permission_request" in allowed:
                        message_id = await whatsapp_api.request_call_permission(
                            row["phone"]
                        )
                        await query(
                            "update portal_case_requests set state='permission',message_id=%s where id=%s and state='preparing'",
                            (message_id, request_id),
                        )
                    else:
                        await self.fail_case(
                            request_id,
                            "WhatsApp cannot request call permission right now. Please try again later.",
                        )
                return
            reserved = await self.reserve(request_id)
            if not reserved:
                return
            claimed = await query(
                """update portal_case_requests set state='calling' where id=%s and runtime_agent_id=%s
                   and state in ('preparing','permission') and expires_at>now() returning id""",
                (request_id, os.environ["VOICE_AGENT_ID"]),
            )
            if claimed:
                context = CallContext(
                    "", row["phone"], str(row["agent_id"]), str(row["customer_id"])
                )
                call_id = await self.dial(request_id, context)
                await query(
                    "update portal_case_requests set call_id=%s where id=%s",
                    (call_id, request_id),
                )
        except Exception:
            logger.exception("Callback failed")
            await self.fail_case(
                request_id,
                "WhatsApp could not place your callback. Please try again shortly.",
            )
        finally:
            if reserved:
                self.reservations.discard(request_id)

    async def dial(self, request_id, context):
        peer = SmallWebRTCConnection(ice_servers=ICE_SERVERS)
        peer.pc.addTransceiver("audio", direction="sendrecv")
        try:
            await peer.pc.setLocalDescription(await peer.pc.createOffer())
            call_id = await whatsapp_api.create_call(
                context.caller_phone, peer.pc.localDescription.sdp, request_id
            )
            self.register(
                CallContext(
                    call_id,
                    context.caller_phone,
                    context.agent_id,
                    context.customer_id,
                    outbound=True,
                ),
                peer,
                outbound=True,
            )
            return call_id
        except Exception:
            await peer.disconnect()
            raise

    async def handle_change(self, value):
        for status in value.get("statuses", []):
            if status.get("status") in ("failed", "REJECTED"):
                await query(
                    """update portal_case_requests set state='failed',error='WhatsApp could not deliver the permission request or call.'
                            where runtime_agent_id=%s and state in ('permission','calling') and (message_id=%s or call_id=%s)""",
                    (os.environ["VOICE_AGENT_ID"], status.get("id"), status.get("id")),
                )
                await self.close_call(status.get("id"))
        for message in value.get("messages", []):
            reply = message.get("interactive", {}).get("call_permission_reply", {})
            if reply.get("response_source") != "user_action" or not message.get(
                "context", {}
            ).get("id"):
                continue
            row = await query(
                """select id from portal_case_requests where phone=%s and message_id=%s and runtime_agent_id=%s
                               and direction='outbound' and state='permission' and expires_at>now()""",
                (
                    message.get("from"),
                    message["context"]["id"],
                    os.environ["VOICE_AGENT_ID"],
                ),
            )
            if row and reply.get("response") == "accept":
                await self.request_case(row["id"])
            elif row and reply.get("response") == "reject":
                await query(
                    "update portal_case_requests set state='declined' where id=%s",
                    (row["id"],),
                )
        for event in value.get("calls", []):
            await self.call_event(event, value)

    async def call_event(self, event, value):
        call_id = event.get("id")
        if not call_id:
            return
        call = self.connections.get(call_id)
        if event.get("event") == "terminate":
            if event.get("status") == "FAILED":
                await query(
                    "update portal_case_requests set state='failed' where call_id=%s and runtime_agent_id=%s and state='calling'",
                    (call_id, os.environ["VOICE_AGENT_ID"]),
                )
            await self.close_call(call_id)
        elif event.get("event") == "connect":
            session = event.get("session", {})
            if session.get("sdp_type") == "answer" and call and call.outbound:
                if not call.ready.is_set():
                    await call.connection.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=session["sdp"], type="answer")
                    )
                    call.answered = True
                    call.ready.set()
            elif session.get("sdp_type") == "offer":
                try:
                    if not await self.reserve(call_id):
                        return

                    async def connected(peer, metadata):
                        self.register(CallContext(call_id, event.get("from", "")), peer)

                    payload = WhatsAppWebhookRequest.model_validate(
                        {
                            "object": "whatsapp_business_account",
                            "entry": [
                                {
                                    "id": "",
                                    "changes": [
                                        {
                                            "field": "calls",
                                            "value": {**value, "calls": [event]},
                                        }
                                    ],
                                }
                            ],
                        }
                    )
                    await self.client.handle_webhook_request(payload, connected)
                except HTTPException:
                    await whatsapp_api.send_call_action(call_id, "reject")
                finally:
                    self.reservations.discard(call_id)

    async def close(self):
        await asyncio.gather(
            *(self.close_call(key, terminate=True) for key in list(self.connections)),
            return_exceptions=True,
        )
        if self.tasks:
            await asyncio.gather(*list(self.tasks), return_exceptions=True)
        await whatsapp_api.close()


@asynccontextmanager
async def lifespan(app):
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=15)
    ) as session:
        app.state.calls.client = WhatsAppClient(
            whatsapp_token=os.environ.get("WHATSAPP_ACCESS_TOKEN")
            or os.environ.get("WHATSAPP_TOKEN", ""),
            phone_number_id=os.environ.get("PHONE_NUMBER_ID", ""),
            session=session,
        )

        async def warmup():
            try:
                await prewarm()
                app.state.voice_ready = True
            except Exception:
                logger.exception("Gemini prewarm failed")
                app.state.voice_startup_error = (
                    "Gemini startup failed. Check the API key and model."
                )

        startup = asyncio.create_task(warmup())
        try:
            yield
        finally:
            app.state.voice_ready = False
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
            await app.state.calls.close()
            await call_state.close()


def create_app():
    app = FastAPI(title="SerendibAI voice runtime", lifespan=lifespan)
    app.state.calls = Calls()
    app.state.voice_ready, app.state.voice_startup_error = False, ""

    @app.get("/")
    def readiness():
        status = (
            "error"
            if app.state.voice_startup_error
            else "ready"
            if app.state.voice_ready
            else "warming_up"
        )
        return {"status": status, "message": "WhatsApp Webhook Server is running"}

    @app.get("/webhook")
    async def verify(request: Request):
        if (
            request.query_params.get("hub.mode") != "subscribe"
            or not os.environ.get("VERIFY_TOKEN")
            or request.query_params.get("hub.verify_token")
            != os.environ["VERIFY_TOKEN"]
        ):
            raise HTTPException(403, "Verification failed")
        return Response(
            request.query_params.get("hub.challenge", ""), media_type="text/plain"
        )

    @app.post("/webhook")
    async def webhook(request: Request):
        body = await request.body()
        secret = os.environ.get("WHATSAPP_APP_SECRET")
        if secret:
            expected = (
                "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            )
            if not hmac.compare_digest(
                expected, request.headers.get("x-hub-signature-256", "")
            ):
                raise HTTPException(403, "Invalid signature")
        data = json.loads(body)
        if data.get("object") != "whatsapp_business_account":
            raise HTTPException(400, "Invalid webhook")
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if value.get("metadata", {}).get(
                    "phone_number_id", os.environ.get("PHONE_NUMBER_ID")
                ) != os.environ.get("PHONE_NUMBER_ID"):
                    raise HTTPException(400, "Wrong business phone")
                if value.get("calls") and not app.state.voice_ready:
                    raise HTTPException(503, "Voice agent unavailable")
                await app.state.calls.handle_change(value)
        return Response("EVENT_RECEIVED")

    @app.post("/cases/call")
    async def callback(request: Request):
        secret = os.environ.get("DEMO_SIGNING_SECRET")
        if not secret or os.environ.get("PORTAL_DEMO_ENABLED") != "1":
            raise HTTPException(404, "Case calling is disabled")
        body = await request.body()
        if not hmac.compare_digest(
            hmac.new(secret.encode(), body, hashlib.sha256).hexdigest(),
            request.headers.get("x-demo-signature", ""),
        ):
            raise HTTPException(403, "Invalid signature")
        try:
            data = json.loads(body)
            valid = (
                re.fullmatch(r"[a-f0-9]{32}", data.get("id", ""))
                and abs(time.time() - data.get("timestamp", 0)) <= 60
            )
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise HTTPException(400, "Invalid callback request")
        if not app.state.voice_ready:
            raise HTTPException(503, "Voice agent unavailable")
        row = await query(
            """update portal_case_requests set state='preparing' where id=%s and phone=%s
                           and runtime_agent_id=%s and direction='outbound' and state='pending' and expires_at>now() returning id""",
            (data["id"], data.get("phone"), os.environ["VOICE_AGENT_ID"]),
        )
        if not row:
            raise HTTPException(409, "Callback already handled or expired")
        await app.state.calls.request_case(data["id"])
        return {"ok": True}

    return app


if __name__ == "__main__":
    from app.store import run_runtime

    run_runtime()
