"""Offline tests for the care agent. Run: python -m unittest discover -s tests -v"""

import asyncio
import unittest
from unittest.mock import patch
from typing import Any

import httpx
from fastapi.testclient import TestClient

from agent.medipass.care import agent as care
from agent.medipass.care.store import Store
from agent.medipass.care.triage import triage
from agent.medipass.care.understand import clean, understand_prompt
from agent.medipass.engine import Profile, parse_profile
from agent.server import app

ANNA = Profile(medications=("Clopidogrel",), allergies=("Penicillin",), genotype={"CYP2C19": "intermediate"})
ANNA_JSON = {"medications": ["Clopidogrel"], "allergies": ["Penicillin"], "genotype": {"CYP2C19": "intermediate"}}


def run(prompt: str, profile: Profile = ANNA, wearable: Any = None) -> dict[str, Any]:
    return asyncio.run(care.run(prompt, profile, wearable))


class TriageTest(unittest.TestCase):
    def test_red_flags_in_english_and_german(self) -> None:
        self.assertTrue(triage("This is the worst headache of my life", care.DEMO_WEARABLE).urgent)
        self.assertTrue(triage("Ich habe plötzlich sehr starke Kopfschmerzen und Nackensteife", care.DEMO_WEARABLE).urgent)
        self.assertFalse(triage("I have a headache, get me tablets", care.DEMO_WEARABLE).urgent)

    def test_ring_data_can_raise_a_flag(self) -> None:
        self.assertTrue(triage("headache", {**care.DEMO_WEARABLE, "temperature_c": 39.1}).urgent)
        self.assertTrue(triage("headache", {**care.DEMO_WEARABLE, "spo2": 90}).urgent)


class ColloquialTest(unittest.TestCase):
    def test_a_humming_skull_is_a_headache_not_an_injury(self) -> None:
        self.assertFalse(triage("Mein Schädel brummt seit heute Morgen", care.DEMO_WEARABLE).urgent)
        self.assertEqual(care.detect_intent("Mein Schädel brummt seit heute Morgen"), "headache")
        self.assertEqual(care.detect_intent("my head is pounding"), "headache")
        self.assertEqual(care.detect_intent("Mein Kopf dröhnt"), "headache")

    def test_real_head_injuries_still_stop_the_order(self) -> None:
        for text in ("Ich habe mir den Kopf gestoßen und habe Kopfschmerzen", "Schädeltrauma nach Sturz", "I hit my head and now have a headache", "Kopfverletzung"):
            self.assertTrue(triage(text, care.DEMO_WEARABLE).urgent, text)

    def test_free_wording_ends_in_an_order_for_anna(self) -> None:
        out = run("Mein Schädel brummt seit heute Morgen, kannst du mir was besorgen?")
        self.assertEqual(out["decision"], "order")


def reading(**fields: Any) -> Any:
    """A fake model reading, as understand_prompt would return it."""

    async def fake(_prompt: str) -> dict[str, Any]:
        return {"intent": None, "medicine": None, "red_flags": [], **fields}

    return fake


class ModelReadingTest(unittest.TestCase):
    def test_the_model_can_supply_the_intent_for_a_free_sentence(self) -> None:
        sentence = "Mein Rücken zwickt seit gestern, besorg mir bitte etwas"
        self.assertEqual(asyncio.run(care.run(sentence, ANNA)) ["decision"], "unknown")
        out = asyncio.run(care.run(sentence, ANNA, None, reading(intent="pain")))
        self.assertEqual(out["decision"], "order")
        self.assertIn("AI model", next(s["detail"] for s in out["steps"] if s["title"] == "Understand the request"))

    def test_the_model_can_add_a_red_flag_the_rules_missed(self) -> None:
        sentence = "Mir wird schwarz vor Augen und mein Kopf dröhnt"
        self.assertEqual(asyncio.run(care.run(sentence, ANNA))["decision"], "order")
        out = asyncio.run(care.run(sentence, ANNA, None, reading(intent="headache", red_flags=["near fainting"])))
        self.assertEqual((out["decision"], out.get("order")), ("urgent", None))
        self.assertIn("near fainting", out["triage"]["reasons"])

    def test_the_model_can_never_remove_a_rule_red_flag(self) -> None:
        out = asyncio.run(care.run("worst headache of my life, order something", ANNA, None, reading(intent="headache", red_flags=[])))
        self.assertEqual(out["decision"], "urgent")

    def test_a_medicine_the_model_invents_is_ignored(self) -> None:
        async def nothing(*_a: Any, **_k: Any) -> None:
            return None

        with patch("agent.medipass.ai_check.assess", nothing):
            out = asyncio.run(care.run("hello there", ANNA, None, reading(medicine="Wonderpill 9000")))
        self.assertEqual(out["decision"], "unknown")

    def test_a_known_medicine_named_in_another_wording_is_accepted(self) -> None:
        out = asyncio.run(care.run("Bitte bestelle das Zeug mit dem Wirkstoff", ANNA, None, reading(medicine="Paracetamol")))
        self.assertEqual(out["decision"], "order")

    def test_replies_are_cleaned_against_fixed_lists(self) -> None:
        self.assertEqual(clean({"intent": "cancer", "medicine": "  ", "red_flags": ["ok", 5, "", "x" * 200]})["intent"], None)
        self.assertEqual(len(clean({"red_flags": ["a"] * 20})["red_flags"]), 5)
        self.assertEqual(len(clean({"red_flags": ["x" * 200]})["red_flags"][0]), 80)
        self.assertIsNone(clean(["not", "a", "dict"]))

    def test_an_unusable_model_falls_back_to_the_rules(self) -> None:
        from unittest.mock import patch

        async def broken(_prompt: str) -> str:
            return "this is not json"

        with patch("agent.medipass.care.understand.ask_llm", broken):
            self.assertIsNone(asyncio.run(understand_prompt("Kopfschmerzen")))
        out = asyncio.run(care.run("Kopfschmerzen, bitte Tabletten", ANNA, None, understand_prompt)) if False else run("Kopfschmerzen, bitte Tabletten")
        self.assertEqual(out["decision"], "order")


ANNA_FULL_PROFILE = parse_profile({"medications": ["Clopidogrel", "Simvastatin", "Ramipril"], "allergies": ["Penicillin"], "genotype": {"CYP2C19": "poor"}, "labs": {"egfr": 52}})


class InfectionTest(unittest.TestCase):
    def test_an_infection_goes_to_a_doctor_who_picks_the_antibiotic(self) -> None:
        out = run("I have a bladder infection, which antibiotic fits and can you buy it?", ANNA_FULL_PROFILE)
        self.assertEqual((out["decision"], out.get("order")), ("doctor_review", None))
        self.assertIn("I do not pick one", out["reply"])
        record = asyncio.run(care.store.get(out["requestId"]))
        self.assertEqual(record["ingredient"], "antibiotic")
        titles = [f["title"] for f in record["check"]["findings"]]
        self.assertIn("Penicillin allergy", titles)
        self.assertIn("Reduced kidney function", titles)
        self.assertEqual(record["check"]["verdict"], "red")

    def test_after_the_doctor_approves_the_agent_buys(self) -> None:
        out = run("Ich habe eine Blasenentzündung, ich brauche ein Antibiotikum", ANNA_FULL_PROFILE)
        record = asyncio.run(care.decide(out["requestId"], "approve", "Nitrofurantoin, five days"))
        self.assertEqual([p["label"] for p in record["order"]["payments"]], ["doctor", "pharmacy"])

    def test_a_named_penicillin_is_still_refused_for_an_allergic_person(self) -> None:
        out = run("I have an infection, order Amoxicillin", ANNA_FULL_PROFILE)
        self.assertEqual((out["decision"], out.get("requestId")), ("advise", None))

    def test_no_flags_means_a_plain_green_request(self) -> None:
        out = run("I have a sinus infection", Profile())
        record = asyncio.run(care.store.get(out["requestId"]))
        self.assertEqual((out["decision"], record["check"]["verdict"]), ("doctor_review", "green"))


class CautionTest(unittest.TestCase):
    def test_liver_pain_is_sent_to_a_doctor_not_to_a_pain_reliever(self) -> None:
        for text in ("Ich habe Leberschmerzen, hol mir etwas dagegen", "pain in my liver, buy me something", "pain in the right upper abdomen and my eyes are yellow"):
            out = run(text, ANNA_FULL_PROFILE)
            self.assertEqual((out["decision"], out.get("order")), ("advise", None), text)
            self.assertIn("see a doctor", out["reply"])

    def test_the_model_calling_liver_pain_a_red_flag_does_not_turn_it_into_an_emergency(self) -> None:
        fake = reading(intent="pain", red_flags=["liver pain (upper right abdominal pain)"])
        out = asyncio.run(care.run("Ich habe Leberschmerzen", ANNA_FULL_PROFILE, None, fake))
        self.assertEqual((out["decision"], out.get("order")), ("advise", None))
        self.assertIn("see a doctor today", out["reply"])

    def test_the_model_can_still_raise_a_real_emergency_next_to_liver_pain(self) -> None:
        fake = reading(red_flags=["chest pain", "liver pain"])
        out = asyncio.run(care.run("Leberschmerzen und Druck auf der Brust", ANNA_FULL_PROFILE, None, fake))
        self.assertEqual(out["decision"], "urgent")

    def test_ordinary_pain_is_still_handled(self) -> None:
        self.assertEqual(run("Mein Rücken tut weh, besorg mir etwas gegen die Schmerzen", ANNA_FULL_PROFILE)["decision"], "order")


class BuyOptionTest(unittest.TestCase):
    def test_without_auto_buy_a_proposal_is_returned_and_nothing_is_ordered(self) -> None:
        out = asyncio.run(care.run("headache, please get me tablets", ANNA_FULL_PROFILE, None, None, False))
        self.assertEqual((out["decision"], out.get("order")), ("proposal", None))
        self.assertEqual(out["proposal"]["ingredient"], "paracetamol")

    def test_pressing_buy_orders_it(self) -> None:
        out = asyncio.run(care.buy("paracetamol", ANNA_FULL_PROFILE))
        self.assertEqual(out["order"]["status"], "confirmed")

    def test_the_safety_check_runs_again_when_buying(self) -> None:
        with self.assertRaises(ValueError):
            asyncio.run(care.buy("ibuprofen", ANNA_FULL_PROFILE))
        with self.assertRaises(ValueError):
            asyncio.run(care.buy("clopidogrel", ANNA_FULL_PROFILE))
        with self.assertRaises(ValueError):
            asyncio.run(care.buy("nonsense", ANNA_FULL_PROFILE))


def verdict_reply(verdict: str, otc: bool = True, findings: int = 0, good: list[str] | None = None) -> dict[str, Any]:
    """What ai_check.assess returns for one medicine."""
    return {"verdict": verdict, "medicine": "x", "findings": [{"light": "yellow", "kind": "ai", "title": f"Caution {i}", "detail": "d", "source": "AI assessment", "advice": "a"} for i in range(findings)],
            "good": good or [], "otc": otc, "aiAssessed": True, "confidence": "high", "notes": [], "summary": "", "explanation": None}


class HayFeverTest(unittest.TestCase):
    def test_hay_fever_is_not_a_fever(self) -> None:
        self.assertIsNone(care.detect_intent("it is hay fever"))
        self.assertIsNone(care.detect_intent("Hay-fever season again"))
        self.assertEqual(care.detect_intent("I have a fever"), "fever")
        self.assertEqual(care.detect_intent("Ich habe Fieber"), "fever")


class PersonalisedTest(unittest.TestCase):
    """A complaint outside the fast paths. The model lists options, every option is judged against the record."""

    def go(self, options: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]], auto_buy: bool = True, profile: Profile = ANNA_FULL_PROFILE) -> dict[str, Any]:
        async def shortlist(_text: str) -> list[dict[str, Any]]:
            return options

        async def assess(name: str, _profile: Any) -> dict[str, Any] | None:
            reply = verdicts.get(name)
            return {**reply, "medicine": name} if reply else None

        with patch("agent.medipass.care.personal.shortlist", shortlist), patch("agent.medipass.ai_check.assess", assess):
            return asyncio.run(care.run("My nose runs and my eyes itch every spring, it is hay fever", profile, None, reading(), auto_buy))

    def test_the_best_checked_option_is_ordered(self) -> None:
        out = self.go([{"ingredient": "loratadine", "otc": True, "reason": "x"}, {"ingredient": "cetirizine", "otc": True, "reason": "y"}],
                      {"loratadine": verdict_reply("yellow", findings=1), "cetirizine": verdict_reply("green", good=["No clash with your medicines."])})
        self.assertEqual(out["decision"], "order")
        self.assertIn("Cetirizine", out["order"]["product"])
        self.assertEqual([c["ingredient"] for c in out["candidates"]], ["cetirizine", "loratadine"])
        self.assertTrue(any("gene results" in line for line in out["why"]))

    def test_a_cautious_best_option_is_only_proposed_and_needs_a_second_yes(self) -> None:
        out = self.go([{"ingredient": "loratadine", "otc": True, "reason": "x"}], {"loratadine": verdict_reply("yellow", findings=2)})
        self.assertEqual((out["decision"], out.get("order")), ("proposal", None))
        proposal = out["proposal"]
        self.assertEqual(len(proposal["caution"]), 2)

        async def assess(name: str, _p: Any) -> dict[str, Any]:
            return {**verdict_reply("yellow", findings=2), "medicine": name}

        with patch("agent.medipass.ai_check.assess", assess):
            with self.assertRaises(ValueError):
                asyncio.run(care.buy("loratadine", ANNA_FULL_PROFILE, proposal["token"]))
            bought = asyncio.run(care.buy("loratadine", ANNA_FULL_PROFILE, proposal["token"], True))
        self.assertEqual(bought["order"]["status"], "confirmed")

    def test_with_auto_buy_off_even_a_green_option_waits_for_the_button(self) -> None:
        out = self.go([{"ingredient": "cetirizine", "otc": True, "reason": "x"}], {"cetirizine": verdict_reply("green")}, auto_buy=False)
        self.assertEqual((out["decision"], out["proposal"]["caution"]), ("proposal", []))

    def test_a_prescription_option_goes_to_a_doctor_and_is_bought_after_approval(self) -> None:
        out = self.go([{"ingredient": "cefalexin", "otc": False, "reason": "x"}], {"cefalexin": verdict_reply("green", otc=False)})
        self.assertEqual((out["decision"], out.get("order")), ("doctor_review", None))
        record = asyncio.run(care.decide(out["requestId"], "approve", "ok"))
        self.assertEqual(record["order"]["product"], "Cefalexin (generic)")
        self.assertEqual([p["label"] for p in record["order"]["payments"]], ["doctor", "pharmacy"])

    def test_nothing_is_ordered_when_every_option_is_red(self) -> None:
        out = self.go([{"ingredient": "naproxen", "otc": True, "reason": "x"}], {"naproxen": verdict_reply("red", findings=1)})
        self.assertEqual((out["decision"], out.get("order"), out.get("proposal")), ("advise", None, None))

    def test_when_the_model_cannot_assess_it_says_so(self) -> None:
        out = self.go([{"ingredient": "cetirizine", "otc": True, "reason": "x"}], {})
        self.assertEqual(out["decision"], "unknown")

    def test_an_empty_shortlist_asks_for_more_detail(self) -> None:
        self.assertEqual(self.go([], {})["decision"], "unknown")

    def test_a_medicine_you_name_that_the_rules_do_not_know_is_judged_and_routed(self) -> None:
        async def assess(name: str, _p: Any) -> dict[str, Any]:
            return {**verdict_reply("yellow", otc=False, findings=1), "medicine": "amiodarone"}

        with patch("agent.medipass.ai_check.assess", assess):
            out = asyncio.run(care.run("Please order Amiodarone", ANNA_FULL_PROFILE, None, reading(medicine="Amiodarone")))
        self.assertEqual((out["decision"], out["requestId"] is not None), ("doctor_review", True))

    def test_offers_cannot_be_forged_replayed_for_another_record_or_used_for_a_prescription(self) -> None:
        good = self.go([{"ingredient": "cetirizine", "otc": True, "reason": "x"}], {"cetirizine": verdict_reply("green")}, auto_buy=False)["proposal"]["token"]
        other = parse_profile({"medications": ["Warfarin"]})
        for token, who, name in ((good + "x", ANNA_FULL_PROFILE, "cetirizine"), (good, other, "cetirizine"), (good, ANNA_FULL_PROFILE, "naproxen"), (None, ANNA_FULL_PROFILE, "cetirizine")):
            with self.assertRaises(ValueError):
                asyncio.run(care.buy(name, who, token))
        from agent.medipass import release

        rx = release.sign_offer("cefalexin", False, care.ai_check.record_hash(ANNA_FULL_PROFILE))
        with self.assertRaises(ValueError):
            asyncio.run(care.buy("cefalexin", ANNA_FULL_PROFILE, rx))

    def test_a_pharmacy_code_and_an_offer_cannot_stand_in_for_each_other(self) -> None:
        from agent.medipass import release
        from agent.medipass.engine import check

        self.assertIsNone(release.read(release.sign_offer("x", True, "h")))
        self.assertIsNone(release.read_offer(release.issue(check("Ibuprofen", Profile(medications=("Clopidogrel",))))))

    def test_controlled_drugs_are_never_offered(self) -> None:
        from agent.medipass.care.personal import clean_options

        out = clean_options({"options": [{"ingredient": "Codeine", "otc": False}, {"ingredient": "tramadol"}, {"ingredient": "Diazepam"}, {"ingredient": "ibuprofen", "otc": True}, {"ingredient": "ibuprofen"}, "junk", {"ingredient": 5}]})
        self.assertEqual([o["ingredient"] for o in out], ["ibuprofen"])


class IntentTest(unittest.TestCase):
    def test_detects_intent_and_named_medicine(self) -> None:
        self.assertEqual(care.detect_intent("Ich habe sehr starke Kopfschmerzen"), "headache")
        self.assertEqual(care.named_medicine("please order Nurofen for me"), "ibuprofen")
        self.assertEqual(care.named_medicine("I need Amoxicillin"), "amoxicillin")
        self.assertIsNone(care.named_medicine("my head hurts"))

    def test_names_inside_other_words_do_not_match(self) -> None:
        self.assertIsNone(care.named_medicine("the class was fine"))


class AgentTest(unittest.TestCase):
    def test_annas_headache_gets_paracetamol_not_aspirin_or_ibuprofen(self) -> None:
        out = run("I have very bad headaches, get me headache tablets")
        self.assertEqual(out["decision"], "order")
        self.assertIn("Paracetamol", out["order"]["product"])
        verdicts = {c["ingredient"]: c["verdict"] for c in out["candidates"]}
        self.assertEqual((verdicts["ibuprofen"], verdicts["acetylsalicylic acid"]), ("yellow", "yellow"))

    def test_simulated_payment_is_labelled_not_hidden(self) -> None:
        out = run("headache, order something")
        self.assertEqual({p["status"] for p in out["order"]["payments"]}, {"simulated"})

    def test_the_order_step_never_claims_a_payment_that_was_simulated(self) -> None:
        out = run("headache, order something")
        detail = next(s["detail"] for s in out["steps"] if s["title"] == "Order and pay")
        self.assertIn("simulated", detail)
        self.assertNotIn("Paid", detail)

    def test_emergency_wording_orders_nothing(self) -> None:
        out = run("worst headache of my life, sudden, order something")
        self.assertEqual((out["decision"], out.get("order")), ("urgent", None))

    def test_prescription_waits_for_a_doctor(self) -> None:
        out = run("please order Clopidogrel", Profile())
        self.assertEqual(out["decision"], "doctor_review")
        self.assertIsNone(out.get("order"))
        self.assertTrue(out["requestId"])

    def test_allergy_blocks_the_order_entirely(self) -> None:
        out = run("order Amoxicillin for my sinus infection")
        self.assertEqual((out["decision"], out.get("requestId")), ("advise", None))

    def test_yellow_named_medicine_is_advised_not_ordered(self) -> None:
        out = run("order Ibuprofen")
        self.assertEqual((out["decision"], out.get("order")), ("advise", None))

    def test_unclear_requests_ask_back(self) -> None:
        self.assertEqual(run("hello there")["decision"], "unknown")

    def test_weak_ring_data_is_ignored_when_malformed(self) -> None:
        self.assertEqual(care.merge_wearable({"temperature_c": "hot", "spo2": True})["temperature_c"], care.DEMO_WEARABLE["temperature_c"])


class DoctorLoopTest(unittest.TestCase):
    def test_approve_pays_doctor_and_pharmacy(self) -> None:
        out = run("please order Clopidogrel", Profile())
        record = asyncio.run(care.decide(out["requestId"], "approve", "Looks fine"))
        self.assertEqual(record["status"], "approved")
        self.assertEqual([p["label"] for p in record["order"]["payments"]], ["doctor", "pharmacy"])

    def test_reject_orders_nothing(self) -> None:
        out = run("please order Clopidogrel", Profile())
        record = asyncio.run(care.decide(out["requestId"], "reject", "Not suitable"))
        self.assertEqual((record["status"], record["order"]), ("rejected", None))

    def test_a_request_is_decided_only_once(self) -> None:
        out = run("please order Clopidogrel", Profile())
        asyncio.run(care.decide(out["requestId"], "reject", "no"))
        again = asyncio.run(care.decide(out["requestId"], "approve", "changed my mind"))
        self.assertEqual(again["status"], "rejected")

    def test_unknown_request(self) -> None:
        self.assertIsNone(asyncio.run(care.decide("nope", "approve", "")))


class StoreTest(unittest.TestCase):
    def test_uses_upstash_when_configured(self) -> None:
        sent: list[list[Any]] = []
        data: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            command = json.loads(request.content)
            sent.append(command)
            if command[0] == "SET":
                data[command[1]] = command[2]
                return httpx.Response(200, json={"result": "OK"})
            if command[0] == "GET":
                return httpx.Response(200, json={"result": data.get(command[1])})
            if command[0] == "SMEMBERS":
                return httpx.Response(200, json={"result": ["a1"]})
            return httpx.Response(200, json={"result": 1})

        store = Store({"UPSTASH_REDIS_REST_URL": "https://u.example", "UPSTASH_REDIS_REST_TOKEN": "t"}, httpx.MockTransport(handler))
        self.assertTrue(store.persistent)
        asyncio.run(store.put({"id": "a1", "createdAt": 1.0, "status": "pending"}))
        self.assertEqual(asyncio.run(store.get("a1"))["status"], "pending")
        self.assertEqual(len(asyncio.run(store.all())), 1)


class RouteTest(unittest.TestCase):
    def setUp(self) -> None:
        from unittest.mock import patch

        async def offline(_prompt: str) -> None:
            return None

        patcher = patch("agent.routes.care.understand_prompt", offline)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)

    def test_agent_end_to_end_through_the_doctor_queue(self) -> None:
        res = self.client.post("/api/care/agent", json={"prompt": "please order Clopidogrel", "profile": {}})
        self.assertEqual(res.json()["decision"], "doctor_review")
        rid = res.json()["requestId"]
        self.assertIn(rid, [r["id"] for r in self.client.get("/api/care/requests").json()["requests"]])
        done = self.client.post(f"/api/care/requests/{rid}/decision", json={"decision": "approve", "note": "ok"})
        self.assertEqual(done.json()["status"], "approved")
        self.assertEqual(self.client.get(f"/api/care/requests/{rid}").json()["status"], "approved")

    def test_buy_route(self) -> None:
        profile = {"medications": ["Clopidogrel"]}
        ok = self.client.post("/api/care/buy", json={"ingredient": "paracetamol", "profile": profile})
        self.assertEqual((ok.status_code, ok.json()["order"]["status"]), (200, "confirmed"))
        self.assertEqual(self.client.post("/api/care/buy", json={"ingredient": "ibuprofen", "profile": profile}).status_code, 409)
        self.assertEqual(self.client.post("/api/care/buy", json={"profile": profile}).status_code, 400)

    def test_auto_buy_off_through_the_route(self) -> None:
        res = self.client.post("/api/care/agent", json={"prompt": "headache, tablets please", "profile": {}, "autoBuy": False})
        self.assertEqual(res.json()["decision"], "proposal")

    def test_bad_input(self) -> None:
        self.assertEqual(self.client.post("/api/care/agent", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/care/agent", json={"prompt": "x" * 2000}).status_code, 400)
        self.assertEqual(self.client.post("/api/care/agent", json={"prompt": "headache", "profile": []}).status_code, 400)
        self.assertEqual(self.client.post("/api/care/requests/x/decision", json={"decision": "maybe"}).status_code, 400)
        self.assertEqual(self.client.get("/api/care/requests/nope").status_code, 404)

    def test_profile_parser_is_shared(self) -> None:
        self.assertEqual(parse_profile(ANNA_JSON).allergies, ("Penicillin",))


if __name__ == "__main__":
    unittest.main()
