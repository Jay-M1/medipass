"""Offline tests for the MediPass rules. Run: python -m unittest discover -s tests -v"""

import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from agent.medipass import release
from agent.medipass.engine import Profile, check, parse_profile, review
from agent.server import app

ANNA = Profile(medications=("Clopidogrel",), genotype={"CYP2C19": "intermediate"})


class EngineTest(unittest.TestCase):
    def test_annas_ibuprofen_is_yellow_with_both_reasons(self) -> None:
        result = check("Ibuprofen", Profile(medications=("Clopidogrel",)))
        self.assertEqual(result.verdict, "yellow")
        self.assertEqual([f.kind for f in result.findings], ["interaction"])

    def test_poor_cyp2c19_makes_clopidogrel_red(self) -> None:
        result = check("Plavix", Profile(genotype={"CYP2C19": "poor"}))
        self.assertEqual((result.verdict, result.medicine), ("red", "clopidogrel"))

    def test_the_worst_finding_wins_and_comes_first(self) -> None:
        result = check("Ibuprofen", Profile(medications=("Warfarin", "Clopidogrel")))
        self.assertEqual(result.verdict, "red")
        self.assertEqual(result.findings[0].light, "red")

    def test_allergy_class_is_red(self) -> None:
        self.assertEqual(check("Amoxicillin", Profile(allergies=("Penicillin",))).verdict, "red")

    def test_condition_rule(self) -> None:
        self.assertEqual(check("Ibuprofen", Profile(conditions=("stomach ulcer",))).verdict, "red")

    def test_known_medicine_without_findings_is_green(self) -> None:
        self.assertEqual(check("Paracetamol", ANNA).verdict, "green")

    def test_unknown_medicine_is_never_green(self) -> None:
        result = check("Madeupol", ANNA)
        self.assertEqual((result.verdict, result.medicine), ("unknown", None))

    def test_unchecked_inputs_are_reported_not_ignored(self) -> None:
        result = check("Ibuprofen", Profile(medications=("Madeupol",), genotype={"CYP2C19": "weird"}))
        self.assertEqual(len(result.notes), 2)

    def test_same_medicine_is_not_an_interaction_with_itself(self) -> None:
        self.assertEqual(check("Ibuprofen", Profile(medications=("Ibuprofen",))).verdict, "green")

    def test_rejects_malformed_profiles(self) -> None:
        with self.assertRaises(ValueError):
            parse_profile({"medications": "clopidogrel"})
        with self.assertRaises(ValueError):
            parse_profile({"genotype": ["CYP2C19"]})


ANNA_BODY = {"profile": {"medications": ["Clopidogrel"], "allergies": ["Penicillin"], "genotype": {"CYP2C19": "intermediate"}}}


class DemoFlowTest(unittest.TestCase):
    """The one-minute pitch demo, step by step."""

    def verdict(self, medicine: str) -> str:
        return check(medicine, parse_profile(ANNA_BODY["profile"])).verdict

    def test_annas_demo(self) -> None:
        self.assertEqual(self.verdict("Ibuprofen"), "yellow")
        self.assertEqual(self.verdict("Omeprazol"), "yellow")
        self.assertEqual(self.verdict("Pantoprazol"), "green")
        self.assertEqual(self.verdict("Amoxicillin"), "red")

    def test_new_gene_rules(self) -> None:
        self.assertEqual(check("Simvastatin", Profile(genotype={"SLCO1B1": "decreased function"})).verdict, "yellow")
        self.assertEqual(check("Warfarin", Profile(genotype={"CYP2C9": "poor"})).verdict, "yellow")


ANNA_FULL = {
    "medications": ["Clopidogrel", "Simvastatin", "Ramipril"],
    "allergies": ["Penicillin"],
    "genotype": {"CYP2C19": "poor", "SLCO1B1": "decreased", "CYP2D6": "normal"},
    "labs": {"egfr": 52},
    "symptoms": ["muscle pain in the legs since the statin"],
}


class AnnaFullTest(unittest.TestCase):
    """The worked example in the team's USP note."""

    profile = parse_profile(ANNA_FULL)

    def test_review_finds_both_treatment_problems_and_the_explanation(self) -> None:
        items = {i["medicine"]: i for i in review(self.profile)["items"]}
        self.assertEqual(items["clopidogrel"]["verdict"], "red")
        self.assertIn("prasugrel", items["clopidogrel"]["findings"][0]["advice"])
        self.assertEqual(items["simvastatin"]["verdict"], "yellow")
        self.assertEqual(len(items["simvastatin"]["mayExplain"]), 1)
        self.assertEqual(items["ramipril"]["verdict"], "green")
        self.assertEqual(review(self.profile)["items"][0]["medicine"], "clopidogrel")

    def test_ibuprofen_hits_three_things_at_once(self) -> None:
        result = check("Ibuprofen", self.profile)
        self.assertEqual(result.verdict, "yellow")
        self.assertEqual(sorted(f.kind for f in result.findings), ["condition", "interaction", "interaction"])
        self.assertTrue(all("Paracetamol" in f.advice for f in result.findings))

    def test_paracetamol_is_green_and_amoxicillin_red(self) -> None:
        self.assertEqual(check("Paracetamol", self.profile).verdict, "green")
        self.assertEqual(check("Amoxicillin", self.profile).verdict, "red")

    def test_normal_kidney_function_is_not_flagged(self) -> None:
        healthy = parse_profile({**ANNA_FULL, "labs": {"egfr": 90}})
        self.assertEqual([f.kind for f in check("Ibuprofen", healthy).findings if f.kind == "condition"], [])

    def test_no_complaint_no_hint(self) -> None:
        quiet = parse_profile({**ANNA_FULL, "symptoms": []})
        self.assertTrue(all(not i["mayExplain"] for i in review(quiet)["items"]))

    def test_a_medicine_is_not_reviewed_against_itself(self) -> None:
        self.assertEqual(review(parse_profile({"medications": ["Ibuprofen"]}))["items"][0]["verdict"], "green")

    def test_bad_labs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_profile({"labs": {"egfr": "low"}})


class ReleaseTest(unittest.TestCase):
    def test_code_shows_warnings_only(self) -> None:
        result = check("Ibuprofen", Profile(medications=("Clopidogrel",), allergies=("Penicillin",)))
        content = release.read(release.issue(result)) or {}
        self.assertEqual((content["verdict"], content["medicine"]), ("yellow", "ibuprofen"))
        self.assertNotIn("clopidogrel", json.dumps(content).lower())
        self.assertNotIn("penicillin", json.dumps(content).lower())

    def test_forged_and_expired_codes_are_rejected(self) -> None:
        code = release.issue(check("Ibuprofen", Profile(medications=("Clopidogrel",))), now=1_000)
        self.assertIsNone(release.read(code, now=1_000 + release.TTL_SECONDS + 1))
        payload, _, signature = code.partition(".")
        self.assertIsNone(release.read(payload + "." + signature[::-1], now=1_001))
        self.assertIsNone(release.read("garbage"))


class RouteTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_free_check_answers(self) -> None:
        res = self.client.post("/api/medipass/check", json={"medicine": "Nurofen", "profile": {"medications": ["Clopidogrel"]}})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["verdict"], "yellow")

    def test_release_roundtrip_and_pharmacy_page(self) -> None:
        issued = self.client.post("/api/medipass/release", json={"medicine": "Ibuprofen", **ANNA_BODY})
        self.assertEqual(issued.status_code, 200)
        code = issued.json()["code"]
        self.assertEqual(self.client.get(f"/api/medipass/release/{code}").json()["verdict"], "yellow")
        page = self.client.get(f"/r/{code}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Higher bleeding risk", page.text)
        self.assertNotIn("Clopidogrel", page.text)
        self.assertEqual(self.client.get("/r/not-a-code").status_code, 404)

    def test_review_route(self) -> None:
        res = self.client.post("/api/medipass/review", json={"profile": ANNA_FULL})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["items"][0]["medicine"], "clopidogrel")
        self.assertEqual(self.client.post("/api/medipass/review", json={}).status_code, 400)

    def test_no_code_for_unknown_medicine(self) -> None:
        self.assertEqual(self.client.post("/api/medipass/release", json={"medicine": "Madeupol"}).status_code, 422)

    def test_explanation_is_optional(self) -> None:
        async def broken(_prompt: str) -> str:
            raise RuntimeError("model down")

        with patch("agent.medipass.explain.ask_llm", broken):
            res = self.client.post("/api/medipass/check", json={"medicine": "Ibuprofen", "explain": True, **ANNA_BODY})
        self.assertEqual(res.status_code, 200)
        self.assertEqual((res.json()["verdict"], res.json()["explanation"]), ("yellow", None))

    def test_explanation_is_added_when_the_model_answers(self) -> None:
        async def fine(_prompt: str) -> str:
            return "Short and calm."

        with patch("agent.medipass.explain.ask_llm", fine):
            res = self.client.post("/api/medipass/check", json={"medicine": "Ibuprofen", "explain": True, **ANNA_BODY})
        self.assertEqual(res.json()["explanation"], "Short and calm.")

    def test_app_page_is_served(self) -> None:
        res = self.client.get("/app")
        self.assertEqual(res.status_code, 200)
        self.assertIn("MediPass", res.text)

    def test_bad_input_is_a_400(self) -> None:
        self.assertEqual(self.client.post("/api/medipass/check", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/medipass/check", json={"medicine": "x", "profile": []}).status_code, 400)
        self.assertEqual(self.client.post("/api/medipass/check", content=b"nope").status_code, 400)


if __name__ == "__main__":
    unittest.main()
