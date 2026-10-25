"""The care agent. It reads the profile and the ring data, checks safety, then orders, asks a doctor, advises or sends to urgent care.

The decisions are rules, not model output. Over-the-counter medicines are ordered only when the traffic light is green and
nothing looks urgent. Prescription medicines always wait for a doctor, who sees the profile and the check.
"""

import os
import re
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Any

from agent.medipass.engine import SEVERITY, CheckResult, Profile, check
from agent.medipass.knowledge import ALIASES

from .catalog import CATALOG, CONSULT_FEE_USDC, INTENTS, Product
from .store import store
from .triage import Triage, triage

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
    return {"ingredient": product.ingredient, "label": product.label, "price_usdc": product.price_usdc, "prescription": product.prescription, "verdict": result.verdict, "warnings": [f.title for f in result.findings]}


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


async def fulfil(ingredient: str, request_id: str, with_consult_fee: bool) -> dict[str, Any]:
    product = CATALOG[ingredient]
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


async def run(prompt: str, profile: Profile, wearable_raw: Any = None, understand: Understand | None = None) -> dict[str, Any]:
    """`understand` is the model's reading of the sentence. Without it only the keyword rules are used."""
    wearable = merge_wearable(wearable_raw)
    reading = await understand(prompt) if understand else None
    steps = [
        {"title": "Read the health record", "detail": f"Medicines {len(profile.medications)}, allergies {len(profile.allergies)}, conditions {len(profile.conditions)}, gene results {len(profile.genotype)}."},
        {"title": "Read the ring", "detail": f"Sleep {wearable['sleep_hours']}h, temperature {wearable['temperature_c']} C, resting heart rate {wearable['resting_hr']}, oxygen {wearable['spo2']} percent."},
    ]
    tri = triage(prompt, wearable)
    if reading and reading["red_flags"]:
        # The model can only add caution. The rule based reasons stay as they are.
        extra = tuple(f for f in reading["red_flags"] if f not in tri.reasons)
        tri = Triage(True, tri.reasons + extra)
    steps.append({"title": "Safety triage", "detail": ("Red flags found, so nothing will be ordered. " + "; ".join(tri.reasons)) if tri.urgent else "No red flags in your request or your ring data."})
    if tri.urgent:
        reply = "This could be serious, so I have not ordered anything. Please contact a doctor or emergency services now (999 in the UK, 112 in the EU). Reason, " + "; ".join(tri.reasons) + "."
        return _result(prompt, None, tri, steps, [], "urgent", reply)

    named, intent = named_medicine(prompt), detect_intent(prompt)
    by_model = False
    if reading:
        if named is None and reading["medicine"]:
            named = named_medicine(reading["medicine"])
            by_model = named is not None
        if intent is None and not named and reading["intent"]:
            intent, by_model = reading["intent"], True
    via = " (understood with the AI model)" if by_model else ""
    notes = _context_notes(wearable) if intent in ("headache", "fever", "pain") else []
    if named and named in CATALOG:
        products = [CATALOG[named]]
        steps.append({"title": "Understand the request", "detail": f"You asked for {named}.{via}"})
    elif intent:
        products = [p for p in CATALOG.values() if intent in p.treats]
        steps.append({"title": "Understand the request", "detail": f"This looks like {intent}.{via} Checking {len(products)} over-the-counter options against your record."})
    else:
        reply = "I could not tell which medicine you need. Tell me a symptom such as a headache or heartburn, or name a medicine, and I will check it against your record."
        return _result(prompt, None, tri, steps, [], "unknown", reply)

    candidates = _rank([_candidate(p, profile) for p in products])
    steps.append({"title": "Check every option", "detail": ", ".join(f"{c['ingredient']} {c['verdict']}" for c in candidates)})
    best = candidates[0]
    product = CATALOG[best["ingredient"]]

    if best["verdict"] == "red":
        reply = f"I will not order {best['ingredient']}. " + "; ".join(best["warnings"]) + ". Please ask your doctor or pharmacist for an alternative."
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
        record["order"] = await fulfil(record["ingredient"], request_id, with_consult_fee=True)
        record["status"] = "approved"
    else:
        record["status"] = "rejected"
    await store.put(record)
    return record
