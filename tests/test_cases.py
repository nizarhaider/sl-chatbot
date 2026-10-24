import asyncio
import os
from contextlib import nullcontext
from unittest.mock import AsyncMock
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from app.prompts import GREETING_TEMPLATE, opening_greeting
from app.agent import agent_settings
from app.store import CallStore
from app import tools, store


def test_case_tools_and_call_records_are_tenant_scoped(monkeypatch):
    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
        with db.transaction(force_rollback=True):
            cases = db.execute(
                "select c.*,a.customer_id from portal_agent_cases c join portal_agents a on a.id=c.agent_id"
            ).fetchall()
            assert len(cases) == 5
            monkeypatch.setattr(tools, "connection", lambda: nullcontext(db))
            monkeypatch.setattr(store, "connection", lambda: nullcontext(db))
            monkeypatch.setenv("VOICE_AGENT_ID", str(cases[0]["runtime_agent_id"]))
            for index, case in enumerate(cases):
                phone = f"1555555000{index}"
                context = tools.CallContext(
                    f"test-case-{index}",
                    phone,
                    str(case["agent_id"]),
                    str(case["customer_id"]),
                )
                assert context.agent_id == str(case["agent_id"])
                assert context.customer_id == str(case["customer_id"])
                config = asyncio.run(tools.DatabaseTools().agent_config(context))
                assert config["is_case"]
                assert config["company_name"] in opening_greeting(config)
                assert "demo" not in opening_greeting(config).lower()
                assert "To speak in English" in opening_greeting(config)
                assert "සිංහල" in opening_greeting(config)
                assert "தமிழ்" in opening_greeting(config)
                assert "How can I help" not in opening_greeting(config)
                assert config["voice"] == "Leda"
                assert agent_settings(config).language is None
                if case["slug"] == "marketing":
                    assert (
                        "You left your number"
                        in agent_settings(config).system_instruction
                    )
                if case["slug"] == "support":
                    order = asyncio.run(
                        tools.DatabaseTools().execute(
                            "search_knowledge", {"query": "WL-1042"}, context
                        )
                    )
                    assert order["ok"] and any(
                        "https://portal.serendibai.lk/cases/order/WL-1042"
                        in row["content"]
                        for row in order["results"]
                    )
                assert config["enabled_tools"]
                result = asyncio.run(
                    tools.DatabaseTools().execute(
                        "create_ticket",
                        {
                            "customer_name": "Demo Caller",
                            "subject": "Routing test",
                            "description": "Transactional test record",
                        },
                        context,
                    )
                )
                assert result["ok"]
                ticket = db.execute(
                    "select agent_id,customer_id from portal_tickets where id=%s",
                    (result["ticket"]["id"],),
                ).fetchone()
                assert str(ticket["agent_id"]) == context.agent_id
                assert str(ticket["customer_id"]) == context.customer_id
                assert not asyncio.run(
                    tools.DatabaseTools().execute("create_order", {}, context)
                )["ok"]
                CallStore().save_call(
                    {
                        "call_id": context.call_id,
                        "caller_phone": phone,
                        "agent_id": context.agent_id,
                        "customer_id": context.customer_id,
                        "started_at": datetime.now(timezone.utc).timestamp(),
                        "events": [
                            {"kind": "gemini_live.usage", "data": {"total_tokens": 30, "usage": {"total_token_count": 30}}},
                            {"kind": "recording.archived", "data": {"bucket": "test-bucket", "key": "test-recording.mp3"}},
                        ],
                    }
                )
                stored = db.execute(
                    "select agent_id,customer_id,tokens,usage,recording_url from portal_calls where id=%s",
                    (f"{context.agent_id}:{context.call_id}",),
                ).fetchone()
                assert str(stored["agent_id"]) == context.agent_id
                assert str(stored["customer_id"]) == context.customer_id
                assert stored["tokens"] == 30
                assert stored["usage"]["requests"] == [{"total_token_count": 30}]
                assert stored["recording_url"] == "s3://test-bucket/test-recording.mp3"


def test_existing_greeting_is_unchanged():
    config = {"name": "Samantha", "company_name": "Talent 360", "is_case": False}
    assert opening_greeting(config) == GREETING_TEMPLATE.format(
        agent_name="Samantha", company_name="Talent 360"
    )
    assert agent_settings(config).language is None


def test_callbacks_dial_immediately_or_after_permission(monkeypatch):
    from app import server

    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
        with db.transaction(force_rollback=True):
            case = db.execute(
                "select c.*,a.customer_id from portal_agent_cases c join portal_agents a on a.id=c.agent_id where c.slug='sales'"
            ).fetchone()
            monkeypatch.setenv("VOICE_AGENT_ID", str(case["runtime_agent_id"]))
            monkeypatch.setattr(server, "connection", lambda: nullcontext(db))
            for index, needs_permission in enumerate((False, True)):
                request_id, phone = f"{100 + index:032x}", f"1202555014{index}"
                db.execute(
                    "insert into portal_case_requests(id,phone,case_slug,runtime_agent_id,direction,state) values(%s,%s,'sales',%s,'outbound','preparing')",
                    (request_id, phone, case["runtime_agent_id"]),
                )
                service = server.Calls()
                allowed, dialed, messages = [not needs_permission], [], []

                async def permission(number):
                    assert number == phone
                    return {
                        "actions": [
                            {
                                "action_name": "start_call"
                                if allowed[0]
                                else "send_call_permission_request",
                                "can_perform_action": True,
                            }
                        ]
                    }

                async def send_permission(number):
                    messages.append(number)
                    return f"permission-{index}"

                async def dial(selected_id, context):
                    dialed.append((selected_id, context.caller_phone, context))
                    return f"outbound-{index}"

                monkeypatch.setattr(server.whatsapp_api, "call_permissions", permission)
                monkeypatch.setattr(
                    server.whatsapp_api, "request_call_permission", send_permission
                )
                monkeypatch.setattr(service, "dial", dial)
                asyncio.run(service.request_case(request_id))
                if needs_permission:
                    assert not dialed
                    assert messages == [phone]
                    reply = {
                        "messages": [
                            {
                                "from": "12025550999",
                                "context": {"id": f"permission-{index}"},
                                "interactive": {
                                    "call_permission_reply": {
                                        "response_source": "user_action",
                                        "response": "accept",
                                    }
                                },
                            }
                        ]
                    }
                    asyncio.run(service.handle_change(reply))
                    assert not dialed
                    reply["messages"][0]["from"] = phone
                    asyncio.run(service.handle_change(reply))
                    assert not dialed
                    allowed[0] = True
                    asyncio.run(service.handle_change(reply))
                    asyncio.run(service.handle_change(reply))
                else:
                    assert not messages
                    asyncio.run(service.request_case(request_id))
                assert len(dialed) == 1
                selected_id, number, context = dialed[0]
                assert (selected_id, number) == (request_id, phone)
                assert context.agent_id == str(case["agent_id"])
                assert context.customer_id == str(case["customer_id"])
                row = db.execute(
                    "select state,call_id from portal_case_requests where id=%s",
                    (request_id,),
                ).fetchone()
                assert row == {"state": "calling", "call_id": f"outbound-{index}"}
                state = store.CallState(persist=False)
                monkeypatch.setattr(server, "call_state", state)
                completed_context = tools.CallContext(f"outbound-{index}", phone, context.agent_id, context.customer_id)
                state.start_call(completed_context)
                peer = AsyncMock()
                service.connections[completed_context.call_id] = server.Call(completed_context, peer, outbound=True, answered=True)
                asyncio.run(service.call_event({"id": completed_context.call_id, "event": "terminate", "status": "COMPLETED"}, {}))
                assert db.execute("select state from portal_case_requests where id=%s", (request_id,)).fetchone()["state"] == "completed"

                db.execute("update portal_case_requests set state='permission',message_id='decline-test' where id=%s", (request_id,))
                asyncio.run(service.handle_change({"messages": [{"from": phone, "context": {"id": "decline-test"}, "interactive": {"call_permission_reply": {"response_source": "user_action", "response": "reject"}}}]}))
                assert db.execute("select state from portal_case_requests where id=%s", (request_id,)).fetchone()["state"] == "declined"
                db.execute("update portal_case_requests set state='preparing',expires_at=now()-interval '1 minute' where id=%s", (request_id,))
                assert asyncio.run(service.case_request(request_id)) is None
                db.execute("update portal_case_requests set expires_at=now()+interval '1 minute' where id=%s", (request_id,))
                with monkeypatch.context() as scoped:
                    scoped.setenv("VOICE_AGENT_ID", "235d1e5e-c41c-43e3-a8d0-db8ee3e6c378")
                    assert asyncio.run(service.case_request(request_id)) is None



def test_callback_endpoint_rejects_unsigned_replayed_and_mismatched_requests(
    monkeypatch,
):
    import hashlib
    import hmac
    import json
    import time
    import httpx
    from app import server

    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
        with db.transaction(force_rollback=True):
            case = db.execute(
                "select * from portal_agent_cases where slug='support'"
            ).fetchone()
            request_id, phone = f"{200:032x}", "12025550149"
            db.execute(
                "insert into portal_case_requests(id,phone,case_slug,runtime_agent_id,direction) values(%s,%s,'support',%s,'outbound')",
                (request_id, phone, case["runtime_agent_id"]),
            )
            monkeypatch.setenv("VOICE_AGENT_ID", str(case["runtime_agent_id"]))
            monkeypatch.setenv("PORTAL_DEMO_ENABLED", "1")
            monkeypatch.setenv("DEMO_SIGNING_SECRET", "test-only-callback-secret")
            monkeypatch.setattr(server, "connection", lambda: nullcontext(db))
            handled = []

            async def request_case(selected_id):
                handled.append(selected_id)

            app = server.create_app()
            monkeypatch.setattr(app.state.calls, "request_case", request_case)
            app.state.voice_ready = True

            async def verify():
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    payload = {
                        "id": request_id,
                        "phone": phone,
                        "timestamp": int(time.time()),
                    }
                    body = json.dumps(payload).encode()
                    assert (
                        await client.post("/cases/call", content=body)
                    ).status_code == 403
                    stale = json.dumps({**payload, "timestamp": int(time.time()) - 120}).encode()
                    signed = hmac.new(b"test-only-callback-secret", stale, hashlib.sha256).hexdigest()
                    assert (await client.post("/cases/call", content=stale, headers={"x-demo-signature": signed})).status_code == 400
                    wrong = json.dumps({**payload, "phone": "12025550999"}).encode()
                    signature = hmac.new(
                        b"test-only-callback-secret", wrong, hashlib.sha256
                    ).hexdigest()
                    assert (
                        await client.post(
                            "/cases/call",
                            content=wrong,
                            headers={"x-demo-signature": signature},
                        )
                    ).status_code == 409
                    signature = hmac.new(
                        b"test-only-callback-secret", body, hashlib.sha256
                    ).hexdigest()
                    assert (
                        await client.post(
                            "/cases/call",
                            content=body,
                            headers={"x-demo-signature": signature},
                        )
                    ).status_code == 200
                    assert (
                        await client.post(
                            "/cases/call",
                            content=body,
                            headers={"x-demo-signature": signature},
                        )
                    ).status_code == 409

            asyncio.run(verify())
            assert handled == [request_id]
