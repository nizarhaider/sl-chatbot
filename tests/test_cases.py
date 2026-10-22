import asyncio
import os
from contextlib import nullcontext
from datetime import datetime, timezone

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

from app.gemini import GREETING_TEMPLATE, opening_greeting
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
                assert "AI " + case["slug"] + " demo" in opening_greeting(config)
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
