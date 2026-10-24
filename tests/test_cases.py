import asyncio
import os
from contextlib import nullcontext
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from app.gemini import GREETING_TEMPLATE, opening_greeting, voice_agent
from app.utility import tools


def test_case_routing_is_verified_once_and_tenant_scoped(monkeypatch):
    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
        with db.transaction(force_rollback=True):
            cases = db.execute("select c.*,a.customer_id from portal_agent_cases c join portal_agents a on a.id=c.agent_id").fetchall()
            assert len(cases) == 5
            monkeypatch.setattr(tools, "connection", lambda: nullcontext(db))
            monkeypatch.setenv("VOICE_AGENT_ID", str(cases[0]["runtime_agent_id"]))
            for index, case in enumerate(cases):
                token = f"{index + 1:032x}"
                phone = f"1555555000{index}"
                db.execute("insert into portal_case_requests(id,phone,case_slug,runtime_agent_id) values(%s,%s,%s,%s)", (token, phone, case["slug"], case["runtime_agent_id"]))
                assert asyncio.run(tools.claim_case(f"unverified-{index}", phone)) is None
                assert asyncio.run(tools.confirm_case("15555559999", token)) is None
                assert asyncio.run(tools.confirm_case(phone, token))["title"] == case["title"]
                context = asyncio.run(tools.claim_case(f"test-case-{index}", phone))
                assert context.agent_id == str(case["agent_id"])
                assert context.customer_id == str(case["customer_id"])
                assert asyncio.run(tools.claim_case("repeat", phone)) is None
                config = asyncio.run(tools.DatabaseTools().agent_config(context))
                assert config["is_case"]
                assert config["company_name"] in opening_greeting(config)
                assert "demo" not in opening_greeting(config).lower()
                assert "To speak in English" in opening_greeting(config)
                assert "සිංහල" in opening_greeting(config)
                assert "தமிழ்" in opening_greeting(config)
                assert "How can I help" not in opening_greeting(config)
                assert config["voice"] == "Leda"
                assert "language_code" not in voice_agent.turn_pipeline._session_config(config)["speech_config"]
                if case["slug"] == "marketing":
                    assert "You left your number" in voice_agent.turn_pipeline._session_config(config)["system_instruction"]
                if case["slug"] == "support":
                    order = asyncio.run(tools.DatabaseTools().execute("search_knowledge", {"query": "WL-1042"}, context))
                    assert order["ok"] and any("https://portal.serendibai.lk/cases/order/WL-1042" in row["content"] for row in order["results"])
                assert config["enabled_tools"]
                result = asyncio.run(tools.DatabaseTools().execute("create_ticket", {"customer_name": "Demo Caller", "subject": "Routing test", "description": "Transactional test record"}, context))
                assert result["ok"]
                ticket = db.execute("select agent_id,customer_id from portal_tickets where id=%s", (result["ticket"]["id"],)).fetchone()
                assert str(ticket["agent_id"]) == context.agent_id
                assert str(ticket["customer_id"]) == context.customer_id
                assert not asyncio.run(tools.DatabaseTools().execute("create_order", {}, context))["ok"]
                tools.CallStore().save_call({"call_id": context.call_id, "caller_phone": phone, "agent_id": context.agent_id, "customer_id": context.customer_id, "started_at": datetime.now(timezone.utc).timestamp()})
                stored = db.execute("select agent_id,customer_id from portal_calls where id=%s", (f"{context.agent_id}:{context.call_id}",)).fetchone()
                assert str(stored["agent_id"]) == context.agent_id
                assert str(stored["customer_id"]) == context.customer_id
            db.execute("update portal_case_requests set state='active',expires_at=now()-interval '1 minute' where id=%s", (token,))
            assert asyncio.run(tools.claim_case("expired", phone)) is None
            db.execute("update portal_case_requests set state='active',expires_at=now()+interval '1 minute' where id=%s", (token,))
            monkeypatch.setenv("VOICE_AGENT_ID", "235d1e5e-c41c-43e3-a8d0-db8ee3e6c378")
            assert asyncio.run(tools.claim_case("other-runtime", phone)) is None


def test_existing_greeting_is_unchanged():
    config = {"name": "Samantha", "company_name": "Talent 360", "is_case": False}
    assert opening_greeting(config) == GREETING_TEMPLATE.format(agent_name="Samantha", company_name="Talent 360")
    assert "language_code" not in voice_agent.turn_pipeline._session_config(config)["speech_config"]


def test_callbacks_dial_immediately_or_after_permission(monkeypatch):
    from app import whatsapp

    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
        with db.transaction(force_rollback=True):
            case = db.execute("select c.*,a.customer_id from portal_agent_cases c join portal_agents a on a.id=c.agent_id where c.slug='sales'").fetchone()
            monkeypatch.setenv("VOICE_AGENT_ID", str(case["runtime_agent_id"]))
            monkeypatch.setattr(whatsapp, "connection", lambda: nullcontext(db))
            for index, needs_permission in enumerate((False, True)):
                request_id, phone = f"{100 + index:032x}", f"1202555014{index}"
                db.execute("insert into portal_case_requests(id,phone,case_slug,runtime_agent_id,direction,state) values(%s,%s,'sales',%s,'outbound','preparing')", (request_id, phone, case["runtime_agent_id"]))
                service = whatsapp.WebRTCService()
                monkeypatch.setattr(whatsapp, "webrtc_service", service)
                allowed, dialed, messages = [not needs_permission], [], []

                async def permission(number):
                    assert number == phone
                    return {"actions": [{"action_name": "start_call" if allowed[0] else "send_call_permission_request", "can_perform_action": True}]}

                async def send_permission(number):
                    messages.append(number)
                    return f"permission-{index}"

                async def dial(selected_id, number, context):
                    dialed.append((selected_id, number, context))
                    return f"outbound-{index}"

                monkeypatch.setattr(whatsapp.whatsapp_api, "call_permissions", permission)
                monkeypatch.setattr(whatsapp.whatsapp_api, "request_call_permission", send_permission)
                monkeypatch.setattr(service, "start_demo", dial)
                asyncio.run(service.request_case(request_id))
                if needs_permission:
                    assert not dialed
                    assert messages == [phone]
                    reply = {"messages": [{"from": "12025550999", "context": {"id": f"permission-{index}"}, "interactive": {"call_permission_reply": {"response_source": "user_action", "response": "accept"}}}]}
                    asyncio.run(whatsapp._handle_change(reply))
                    assert not dialed
                    reply["messages"][0]["from"] = phone
                    asyncio.run(whatsapp._handle_change(reply))
                    assert not dialed
                    allowed[0] = True
                    asyncio.run(whatsapp._handle_change(reply))
                    asyncio.run(whatsapp._handle_change(reply))
                else:
                    assert not messages
                    asyncio.run(service.start_case(request_id))
                assert len(dialed) == 1
                selected_id, number, context = dialed[0]
                assert (selected_id, number) == (request_id, phone)
                assert context.agent_id == str(case["agent_id"])
                assert context.customer_id == str(case["customer_id"])
                row = db.execute("select state,call_id from portal_case_requests where id=%s", (request_id,)).fetchone()
                assert row == {"state": "calling", "call_id": f"outbound-{index}"}
                asyncio.run(whatsapp._handle_change({"calls": [{"id": f"outbound-{index}", "event": "terminate", "status": "COMPLETED"}]}))
                assert db.execute("select state from portal_case_requests where id=%s", (request_id,)).fetchone()["state"] == "completed"


def test_callback_endpoint_rejects_unsigned_replayed_and_mismatched_requests(monkeypatch):
    import hashlib
    import hmac
    import json
    import time
    import httpx
    from app import whatsapp

    load_dotenv(".env")
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as db:
        with db.transaction(force_rollback=True):
            case = db.execute("select * from portal_agent_cases where slug='support'").fetchone()
            request_id, phone = f"{200:032x}", "12025550149"
            db.execute("insert into portal_case_requests(id,phone,case_slug,runtime_agent_id,direction) values(%s,%s,'support',%s,'outbound')", (request_id, phone, case["runtime_agent_id"]))
            monkeypatch.setenv("VOICE_AGENT_ID", str(case["runtime_agent_id"]))
            monkeypatch.setenv("PORTAL_DEMO_ENABLED", "1")
            monkeypatch.setenv("DEMO_SIGNING_SECRET", "test-only-callback-secret")
            monkeypatch.setattr(whatsapp, "connection", lambda: nullcontext(db))
            handled = []

            async def request_case(selected_id):
                handled.append(selected_id)

            monkeypatch.setattr(whatsapp.webrtc_service, "request_case", request_case)
            app = whatsapp.create_app()
            app.state.voice_ready = True

            async def verify():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                    payload = {"id": request_id, "phone": phone, "timestamp": int(time.time())}
                    body = json.dumps(payload).encode()
                    assert (await client.post("/cases/call", content=body)).status_code == 403
                    wrong = json.dumps({**payload, "phone": "12025550999"}).encode()
                    signature = hmac.new(b"test-only-callback-secret", wrong, hashlib.sha256).hexdigest()
                    assert (await client.post("/cases/call", content=wrong, headers={"x-demo-signature": signature})).status_code == 409
                    signature = hmac.new(b"test-only-callback-secret", body, hashlib.sha256).hexdigest()
                    assert (await client.post("/cases/call", content=body, headers={"x-demo-signature": signature})).status_code == 200
                    assert (await client.post("/cases/call", content=body, headers={"x-demo-signature": signature})).status_code == 409

            asyncio.run(verify())
            assert handled == [request_id]
