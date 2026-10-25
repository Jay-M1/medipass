"""The care agent. It reads the profile and the ring data, checks safety, then orders, asks a doctor, advises or sends to urgent care.

The decisions are rules, not model output. Over-the-counter medicines are ordered only when the traffic light is green and
nothing looks urgent. Prescription medicines always wait for a doctor, who sees the profile and the check.
"""

import asyncio
import os
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agent.medipass import ai_check, release
from agent.medipass.engine import SEVERITY, CheckResult, Profile, check
from agent.medipass.knowledge import ALIASES

from . import personal
from .catalog import CATALOG, CONSULT_FEE_USDC, INTENTS, Product
from .store import store
from .triage import Triage, cautions, triage

# Made up demo readings from a smart ring. A real build would read them from the ring's app or health platform.
DEMO_WEARABLE: dict[str, Any] = {"device": "Demo ring", "resting_hr": 68, "hrv_ms": 41, "sleep_hours": 4.8, "temperature_c": 36.8, "spo2": 97, "steps": 3200}
NUMERIC_WEARABLE = ("resting_hr", "hrv_ms", "sleep_hours", "temperature_c", "spo2", "steps")
EXPLORER = "https://explorer.solana.com/tx/{}?cluster=devnet"
NOT_CHECKED = "Only the data in your profile was checked. MediPass does not replace a pharmacist or doctor."


def merge_wearable(raw: Any) -> dict[str, Any]:
    merged = dict(DEMO_WEARABLE)
    if isinstance(raw, dict):
        for key in NUMERIC_WEARABLE:
            if isinstance(raw.get(key), (int, float)) and not isinstance(raw.get(key), bool):
                merged[key] = raw[key]
    return merged


def detect_intent(prompt: str) -> str | None:
    text = prompt.lower()
    return next((name for name, pattern in INTENTS if re.search(pattern, text)), None)


# People describe a medicine by what it is for. These words are resolved against the medicines the person already takes.
CLASS_WORDS: dict[str, tuple[str, ...]] = {
    "blood thinner": ("blood thinner", "blood-thinner", "blutverdünner", "blutverduenner", "gerinnungshemmer", "anticoagulant", "antiplatelet", "thrombozytenaggregationshemmer"),
    "cholesterol": ("cholesterol", "cholesterin", "statin"),
    "blood pressure": ("blood pressure", "blutdruck", "hypertension", "bluthochdruck"),
}
CLASS_MEMBERS: dict[str, frozenset[str]] = {
    "blood thinner": frozenset({"clopidogrel", "warfarin", "ticagrelor", "prasugrel", "apixaban", "rivaroxaban", "phenprocoumon", "acetylsalicylic acid"}),
    "cholesterol": frozenset({"simvastatin", "atorvastatin", "rosuvastatin", "pravastatin"}),
    "blood pressure": frozenset({"ramipril", "lisinopril", "amlodipine", "bisoprolol", "metoprolol", "candesartan", "losartan", "valsartan"}),
}


def class_medicine(text: str, profile: Profile) -> str | None:
    """"My blood thinner" becomes the blood thinner on this person's own list, or None if they take none."""
    lowered = text.lower()
    for kind, words in CLASS_WORDS.items():
        if any(w in lowered for w in words):
            for taken in profile.medications:
                name = ALIASES.get(taken.strip().lower(), taken.strip().lower())
                if name in CLASS_MEMBERS[kind]:
                    return name
    return None


def named_medicine(prompt: str) -> str | None:
    """An ingredient the person asked for by name, taking the longest matching name."""
    text = prompt.lower()
    for alias in sorted(ALIASES, key=len, reverse=True):
        if re.search(rf"(?<![a-zäöü]){re.escape(alias)}(?![a-zäöü])", text):
            return ALIASES[alias]
    return None


def _context_notes(wearable: dict[str, Any]) -> list[str]:
    notes = []
    if wearable["sleep_hours"] < 6:
        notes.append(f"Your ring shows {wearable['sleep_hours']:.1f} hours of sleep. Short sleep and little water can trigger headaches.")
    if wearable["temperature_c"] >= 37.8:
        notes.append(f"Your ring measured {wearable['temperature_c']:.1f} C, which can be a mild fever.")
    return notes


def _candidate(product: Product, profile: Profile) -> dict[str, Any]:
    result = check(product.ingredient, profile)
    return {"ingredient": product.ingredient, "label": product.label, "price_usdc": product.price_usdc, "prescription": product.prescription, "verdict": result.verdict,
            "warnings": [f.title for f in result.findings], "details": [f.detail for f in result.findings], "advice": [f.advice for f in result.findings if f.advice]}


def _rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {c: i for i, c in enumerate(CATALOG)}
    return sorted(candidates, key=lambda c: (SEVERITY.get(c["verdict"], 3), order[c["ingredient"]]))


def _summary(profile: Profile) -> dict[str, Any]:
    return {"medications": list(profile.medications), "allergies": list(profile.allergies), "conditions": list(profile.conditions), "genotype": profile.genotype}


async def settle(label: str, address: str | None, amount_usdc: float, memo: str) -> dict[str, Any]:
    """Pays on Solana devnet, or says plainly that it only simulated the payment."""
    base = {"label": label, "amount_usdc": amount_usdc, "memo": memo}
    if not address:
        return {**base, "status": "simulated", "reason": f"No receiving address is configured for the {label}."}
    from agent.paidapi.buyer import get_spending_tracker
    from agent.payments.config import PaymentsConfigError
    from agent.payments.transfer import TransferError, load_agent_keypair, pay_usdc

    try:
        signature = await pay_usdc(load_agent_keypair(), to=address, amount_usdc=amount_usdc, memo=memo, guard=get_spending_tracker())
    except PaymentsConfigError as err:
        return {**base, "status": "simulated", "reason": str(err)}
    except TransferError as err:
        return {**base, "status": "failed", "reason": str(err)}
    return {**base, "status": "paid", "signature": signature, "explorer": EXPLORER.format(signature)}


async def fulfil(ingredient: str, request_id: str, with_consult_fee: bool, product: Product | None = None) -> dict[str, Any]:
    product = product or CATALOG[ingredient]
    payments = []
    if with_consult_fee:
        payments.append(await settle("doctor", os.environ.get("MEDIPASS_DOCTOR_ADDRESS"), CONSULT_FEE_USDC, f"medipass:review:{request_id}"))
    payments.append(await settle("pharmacy", os.environ.get("MEDIPASS_PHARMACY_ADDRESS"), product.price_usdc, f"medipass:order:{request_id}"))
    paid = all(p["status"] in ("paid", "simulated") for p in payments)
    return {"id": request_id, "product": product.label, "status": "confirmed" if paid else "payment_failed", "delivery": "Home delivery tomorrow, 10:00 to 12:00 (demo)", "payments": payments}


def _payment_summary(payments: list[dict[str, Any]]) -> str:
    words = {"paid": "Paid on Solana devnet", "simulated": "Payment simulated, no real transfer", "failed": "Payment failed"}
    return "; ".join(f"{words[p['status']]} ({p['label']}, {p['amount_usdc']} USDC)" for p in payments) + "."


def _result(prompt: str, intent: str | None, tri: Any, steps: list[dict[str, str]], candidates: list[dict[str, Any]], decision: str, reply: str, **extra: Any) -> dict[str, Any]:
    return {"prompt": prompt, "intent": intent, "triage": {"urgent": tri.urgent, "reasons": list(tri.reasons)}, "steps": steps, "candidates": candidates, "decision": decision, "reply": reply, **extra}


Understand = Callable[[str], Awaitable[dict[str, Any] | None]]


def _antibiotic_findings(profile: Profile) -> list[dict[str, str]]:
    """What the record says that matters when a doctor picks an antibiotic. MediPass shows flags and never picks the antibiotic."""
    findings: list[dict[str, str]] = []
    allergies = " ".join(profile.allergies).lower()
    if "penicillin" in allergies or "amoxicillin" in allergies:
        findings.append({"light": "red", "kind": "allergy", "title": "Penicillin allergy", "detail": "Allergy record. Penicillins such as amoxicillin are not suitable.", "source": "Allergy record", "advice": "Choose an antibiotic from another group."})
    egfr = profile.labs.get("egfr")
    if egfr is not None and egfr < 60:
        findings.append({"light": "yellow", "kind": "condition", "title": "Reduced kidney function", "detail": f"eGFR {egfr:g}. Some antibiotics need a lower dose.", "source": "Lab value", "advice": "Check the dose against kidney function."})
    if profile.medications:
        findings.append({"light": "yellow", "kind": "interaction", "title": "Other medicines in use", "detail": "Takes " + ", ".join(profile.medications) + ". Check for interactions with the antibiotic you choose.", "source": "Medication list", "advice": "Check interactions."})
    return findings


async def _ask_doctor_about_antibiotic(prompt: str, profile: Profile, wearable: dict[str, Any], tri: Any, steps: list[dict[str, str]], notes: list[str]) -> dict[str, Any]:
    product = CATALOG["antibiotic"]
    findings = _antibiotic_findings(profile)
    verdict = "red" if any(f["light"] == "red" for f in findings) else "yellow" if findings else "green"
    request_id = secrets.token_hex(4)
    record = {"id": request_id, "status": "pending", "createdAt": time.time(), "prompt": prompt, "ingredient": product.ingredient, "product": product.label, "profile": _summary(profile), "wearable": wearable,
              "check": {"verdict": verdict, "medicine": "antibiotic", "findings": findings, "notes": ["The doctor decides whether an antibiotic is needed and which one."]}, "decision": None, "order": None}
    await store.put(record)
    flags = [f["title"].lower() for f in findings[:2]]
    steps.append({"title": "Antibiotics need a doctor", "detail": "Whether you need one and which one is the doctor's decision. Your record and its flags were sent for a quick yes or no."})
    reply = ("Antibiotics need a doctor, who first decides whether you need one at all and which one. I do not pick one for you. I sent your record to a doctor"
             + (f" with these flags, {', '.join(flags)}" if flags else "") + ". Once they approve, I pay the doctor and the pharmacy and arrange delivery.")
    return _result(prompt, "infection", tri, steps, [], "doctor_review", reply, requestId=request_id, notes=notes)


async def _personalised(prompt: str, unknown_named: str | None, profile: Profile, wearable: dict[str, Any], tri: Any, steps: list[dict[str, str]], notes: list[str], auto_buy: bool, via: str) -> dict[str, Any]:
    """A complaint outside the fast paths, or a medicine the rules do not know. The model lists the standard options and every one
    is judged against this record. The best one is ordered, proposed or sent to a doctor. Demo only, nothing is ordered for real."""
    if unknown_named:
        options = [{"ingredient": re.sub(r"\s+", " ", unknown_named).strip().lower()[:60], "otc": None, "reason": "You asked for it."}]
        steps.append({"title": "Understand the request", "detail": f"You asked for {unknown_named}, which is not in our rule set.{via}"})
    else:
        options = await personal.shortlist(prompt)
        if not options:
            reply = "I could not tell which medicine would help. Describe the complaint a little more, or see a doctor if it is not going away."
            return _result(prompt, None, tri, steps, [], "unknown", reply, notes=notes)
        steps.append({"title": "Standard options for this complaint", "detail": ", ".join(o["ingredient"] for o in options) + f".{via}"})
    evaluated = list(await asyncio.gather(*(personal.evaluate(o, profile) for o in options)))
    order = sorted(range(len(evaluated)), key=lambda i: (SEVERITY.get(evaluated[i]["verdict"], 3), len(evaluated[i]["findings"]), i))
    candidates = [{**evaluated[i], "prescription": evaluated[i]["otc"] is not True} for i in order]
    steps.append({"title": "Check every option against your record", "detail": ", ".join(f"{c['ingredient']} {c['verdict']}" for c in candidates)})
    usable = [c for c in candidates if c["verdict"] in ("green", "yellow")]
    if not usable:
        unknown = all(c["verdict"] == "unknown" for c in candidates)
        reply = ("I could not assess that medicine right now. Try again, or ask a pharmacist." if unknown
                 else "None of the usual options is safe for your record. " + "; ".join(f"{c['ingredient']}, {c['warnings'][0]}" for c in candidates if c["warnings"])[:300] + ". Please ask your doctor or pharmacist.")
        return _result(prompt, None, tri, steps, candidates, "unknown" if unknown else "advise", reply, notes=notes + [NOT_CHECKED])

    best = usable[0]
    otc = best["otc"] is True
    reasons = personal.why(best, profile)
    caution = best["warnings"] if best["verdict"] == "yellow" else []
    product = personal.product_for(best["ingredient"], otc, best["price_usdc"])
    extra: dict[str, Any] = {"why": reasons, "notes": notes + [NOT_CHECKED]}
    if not otc:
        request_id = secrets.token_hex(4)
        record = {"id": request_id, "status": "pending", "createdAt": time.time(), "prompt": prompt, "ingredient": best["ingredient"], "product": product.label, "priceUsdc": product.price_usdc, "profile": _summary(profile), "wearable": wearable,
                  "check": {"verdict": best["verdict"], "medicine": best["ingredient"], "findings": best["findings"], "notes": ["Chosen by MediPass for the complaint and checked against the record. The doctor decides."]}, "decision": None, "order": None}
        await store.put(record)
        steps.append({"title": "Prescription needed", "detail": "Sent to a doctor for a quick yes or no. No appointment is needed."})
        reply = f"The best match for your record is {best['ingredient']}, and it needs a doctor's approval. I sent your record and the check to a doctor. Once they approve, I pay the doctor and the pharmacy and arrange delivery."
        return _result(prompt, None, tri, steps, candidates, "doctor_review", reply, requestId=request_id, **extra)
    if best["verdict"] == "green" and auto_buy:
        request_id = secrets.token_hex(4)
        order = await fulfil(best["ingredient"], request_id, with_consult_fee=False, product=product)
        steps.append({"title": "Order and pay", "detail": f"Ordered {product.label}. {_payment_summary(order['payments'])}"})
        reply = f"I ordered {product.label}. It is the best match for your complaint and your record."
        return _result(prompt, None, tri, steps, candidates, "order", reply, order=order, **extra)
    token = release.sign_offer(best["ingredient"], True, ai_check.record_hash(profile))
    steps.append({"title": "Waiting for you", "detail": "This is a proposal. Nothing is ordered until you press buy."})
    reply = (f"I suggest {product.label}. It is the best match for your complaint and your record." if not caution
             else f"The best match is {product.label}, but there are cautions, {'; '.join(caution[:2])}. Read them before you decide.")
    proposal = {"ingredient": best["ingredient"], "label": product.label, "price_usdc": product.price_usdc, "caution": caution, "token": token}
    return _result(prompt, None, tri, steps, candidates, "proposal", reply, proposal=proposal, **extra)


async def run(prompt: str, profile: Profile, wearable_raw: Any = None, understand: Understand | None = None, auto_buy: bool = True) -> dict[str, Any]:
    """`understand` is the model's reading of the sentence. Without it only the keyword rules are used.

    With auto_buy off, a safe over-the-counter pick comes back as a proposal and nothing is ordered until buy() is called."""
    wearable = merge_wearable(wearable_raw)
    reading = await understand(prompt) if understand else None
    steps = [
        {"title": "Read the health record", "detail": f"Medicines {len(profile.medications)}, allergies {len(profile.allergies)}, conditions {len(profile.conditions)}, gene results {len(profile.genotype)}."},
        {"title": "Read the ring", "detail": f"Sleep {wearable['sleep_hours']}h, temperature {wearable['temperature_c']} C, resting heart rate {wearable['resting_hr']}, oxygen {wearable['spo2']} percent."},
    ]
    tri = triage(prompt, wearable)
    care = cautions(prompt)
    if reading and reading["red_flags"]:
        # The model can only add caution. The rule based reasons stay as they are. Complaints that the cautions below already
        # handle (pain around the liver) go to a doctor, not to the emergency line, so the model's wording for them is skipped.
        handled = re.compile(r"liver|gall|abdom|belly|jaundice|yellow|urine", re.I) if care else None
        extra = tuple(f for f in reading["red_flags"] if f not in tri.reasons and not (handled and handled.search(f)))
        if extra:
            tri = Triage(True, tri.reasons + extra)
    steps.append({"title": "Safety triage", "detail": ("Red flags found, so nothing will be ordered. " + "; ".join(tri.reasons)) if tri.urgent else "No red flags in your request or your ring data."})
    if tri.urgent:
        reply = "This could be serious, so I have not ordered anything. Please contact a doctor or emergency services now (999 in the UK, 112 in the EU). Reason, " + "; ".join(tri.reasons) + "."
        return _result(prompt, None, tri, steps, [], "urgent", reply)

    if care:
        steps.append({"title": "Caution", "detail": "; ".join(care).capitalize() + ". A medicine should not be picked for this without a doctor."})
        reply = ("I did not order anything. " + "; ".join(care).capitalize() + " should be looked at by a doctor first, and common pain relievers such as paracetamol can strain the liver, so I will not choose one for you. "
                 "Please see a doctor today. If the pain is severe or your skin or eyes turn yellow, call 999 or 112.")
        return _result(prompt, None, tri, steps, [], "advise", reply, notes=[NOT_CHECKED])

    named, intent = named_medicine(prompt), detect_intent(prompt)
    by_model = False
    if reading:
        if named is None and reading["medicine"]:
            named = named_medicine(reading["medicine"])
            by_model = named is not None
        if intent is None and not named and reading["intent"]:
            intent, by_model = reading["intent"], True
    if named is None:
        # "my blood thinner" means the one this person already takes
        named = class_medicine(prompt, profile) or (class_medicine(reading["medicine"], profile) if reading and reading.get("medicine") else None)
    via = " (understood with the AI model)" if by_model else ""
    notes = _context_notes(wearable) if intent in ("headache", "fever", "pain") else []
    if reading and reading.get("medicine") and named is None:
        return await _personalised(prompt, reading["medicine"], profile, wearable, tri, steps, notes, auto_buy, via)
    if intent == "infection" and not (named and named in CATALOG):
        steps.append({"title": "Understand the request", "detail": f"This looks like an infection.{via}"})
        return await _ask_doctor_about_antibiotic(prompt, profile, wearable, tri, steps, notes)
    if named and named in CATALOG:
        products = [CATALOG[named]]
        steps.append({"title": "Understand the request", "detail": f"You asked for {named}.{via}"})
    elif intent:
        products = [p for p in CATALOG.values() if intent in p.treats]
        steps.append({"title": "Understand the request", "detail": f"This looks like {intent}.{via} Checking {len(products)} over-the-counter options against your record."})
    elif reading is not None:
        return await _personalised(prompt, None, profile, wearable, tri, steps, notes, auto_buy, via)
    else:
        reply = "I could not tell which medicine you need. Tell me a symptom such as a headache or heartburn, or name a medicine, and I will check it against your record."
        return _result(prompt, None, tri, steps, [], "unknown", reply)

    candidates = _rank([_candidate(p, profile) for p in products])
    steps.append({"title": "Check every option", "detail": ", ".join(f"{c['ingredient']} {c['verdict']}" for c in candidates)})
    best = candidates[0]
    product = CATALOG[best["ingredient"]]

    if best["verdict"] == "red":
        reply = f"I will not order {best['ingredient']}. {best['warnings'][0]}. {best['details'][0]} " + (best["advice"][0] if best["advice"] else "Please ask your doctor or pharmacist for an alternative.")
        if best["ingredient"] in {taken.strip().lower() for taken in profile.medications} or any(best["ingredient"] == (ALIASES.get(taken.strip().lower())) for taken in profile.medications):
            reply += " Do not stop a medicine you already take on your own. Talk to your doctor first."
        return _result(prompt, intent, tri, steps, candidates, "advise", reply, notes=notes)

    if product.prescription:
        request_id = secrets.token_hex(4)
        record = {"id": request_id, "status": "pending", "createdAt": time.time(), "prompt": prompt, "ingredient": product.ingredient, "product": product.label, "profile": _summary(profile), "wearable": wearable, "check": check(product.ingredient, profile).to_dict(), "decision": None, "order": None}
        await store.put(record)
        steps.append({"title": "Prescription needed", "detail": "Sent to a doctor for a quick yes or no. No appointment is needed."})
        reply = f"{product.label} needs a doctor's approval. I sent your record and the safety check to a doctor for a quick review. Once they approve, I pay the doctor and the pharmacy and arrange delivery."
        return _result(prompt, intent, tri, steps, candidates, "doctor_review", reply, requestId=request_id, notes=notes)

    if best["verdict"] == "yellow":
        reply = f"I did not order {best['ingredient']} on my own, because of this. " + "; ".join(best["warnings"]) + ". A pharmacist or doctor should confirm first."
        return _result(prompt, intent, tri, steps, candidates, "advise", reply, notes=notes)

    if not auto_buy:
        steps.append({"title": "Waiting for you", "detail": "This is a proposal. Nothing is ordered until you press buy."})
        proposal = {"ingredient": best["ingredient"], "label": product.label, "price_usdc": product.price_usdc}
        reply = f"I suggest {product.label}. It is the best match for your record. Press buy and I will order it and pay."
        return _result(prompt, intent, tri, steps, candidates, "proposal", reply, proposal=proposal, notes=notes + [NOT_CHECKED])

    request_id = secrets.token_hex(4)
    order = await fulfil(best["ingredient"], request_id, with_consult_fee=False)
    steps.append({"title": "Order and pay", "detail": f"Ordered {product.label}. {_payment_summary(order['payments'])}"})
    reply = f"I ordered {product.label}. It is the best match for your record" + (f", and the other options had warnings ({', '.join(c['ingredient'] for c in candidates[1:] if c['warnings'])})." if any(c["warnings"] for c in candidates[1:]) else ".")
    return _result(prompt, intent, tri, steps, candidates, "order", reply, order=order, notes=notes + [NOT_CHECKED])


async def decide(request_id: str, decision: str, note: str) -> dict[str, Any] | None:
    record = await store.get(request_id)
    if record is None:
        return None
    if record["status"] != "pending":
        return record
    record["decision"] = {"verdict": decision, "note": note[:300], "at": time.time()}
    if decision == "approve":
        product = None if record["ingredient"] in CATALOG else Product(record["ingredient"], record["product"], record.get("priceUsdc", personal.DEMO_PRICE_USDC), True, ())
        record["order"] = await fulfil(record["ingredient"], request_id, with_consult_fee=True, product=product)
        record["status"] = "approved"
    else:
        record["status"] = "rejected"
    await store.put(record)
    return record


async def buy(ingredient: str, profile: Profile, token: str | None = None, accept_caution: bool = False) -> dict[str, Any]:
    """Orders an over-the-counter medicine after the person pressed buy. The safety check runs again here, so a client cannot skip it.
    A medicine outside the fixed catalog needs a signed offer that the agent made for this very record."""
    product = CATALOG.get(ingredient)
    if product is not None:
        if product.prescription:
            raise ValueError("Only over-the-counter medicines can be bought directly. A doctor has to approve the rest.")
        verdict = check(ingredient, profile).verdict
    else:
        offer = release.read_offer(token or "")
        if not offer or offer["ingredient"] != ingredient or offer["recordHash"] != ai_check.record_hash(profile):
            raise ValueError("Not ordered. This offer is not valid any more. Ask the agent again.")
        if not offer["otc"]:
            raise ValueError("Only over-the-counter medicines can be bought directly. A doctor has to approve the rest.")
        verdict = (await personal.evaluate({"ingredient": ingredient, "otc": True}, profile))["verdict"]
        product = personal.product_for(ingredient, True)
    if verdict != "green" and not (verdict == "yellow" and accept_caution):
        raise ValueError(f"Not ordered. The safety check for {ingredient} is {verdict}." + (" Confirm that you want it anyway." if verdict == "yellow" else ""))
    request_id = secrets.token_hex(4)
    order = await fulfil(ingredient, request_id, with_consult_fee=False, product=product)
    return {"order": order, "reply": f"Ordered {product.label}. {_payment_summary(order['payments'])}"}
