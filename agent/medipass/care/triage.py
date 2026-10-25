"""Red-flag triage that runs before anything is ordered.

The agent must never treat a possible emergency as a shopping task. Any red flag stops the order and sends the person to urgent care.
Keywords cover English and German. It is deliberately cautious, not a diagnosis.
"""

import re
from dataclasses import dataclass, field
from typing import Any

RED_FLAGS: tuple[tuple[str, str], ...] = (
    (r"worst headache|schlimmste[nrs]? kopfschmerz", "the worst headache of your life"),
    (r"thunderclap|sudden(ly)?\s+(severe|intense|explosive)|ploetzlich|plötzlich", "a sudden, very severe onset"),
    (r"stiff neck|nackensteif", "a stiff neck"),
    (r"double vision|vision (loss|problem)|can'?t see|sehstoerung|sehstörung|doppelbild", "vision problems"),
    (r"confus|verwirr|slurred|sprachstoerung|sprachstörung|trouble speaking", "confusion or speech problems"),
    (r"numb|taub|weakness (on|in) (one|my)|laehmung|lähmung|droop", "numbness or weakness"),
    (r"head injury|hit my head|banged my head|kopfverletzung|kopf angeschlagen|kopf gestossen|kopf gestoßen|schädel-?hirn|schaedel-?hirn|schädel(trauma|bruch|prellung)|schaedel(trauma|bruch|prellung)|schädel (angeschlagen|geprellt|verletzt)", "a recent head injury"),
    (r"seizure|krampf|faint|ohnmacht|bewusstlos", "a seizure or fainting"),
    (r"chest pain|brustschmerz|shortness of breath|atemnot", "chest pain or trouble breathing"),
    (r"pregnan|schwanger", "pregnancy"),
)

# Not emergencies, but a pain reliever should not be picked for these without a doctor.
CAUTIONS: tuple[tuple[str, str], ...] = (
    (r"liver|leber|upper right (belly|abdomen)|right upper (belly|abdomen)|rechter oberbauch|oberbauch rechts|gallen|gall bladder|gallbladder", "pain around the liver or gallbladder"),
    (r"yellow (skin|eyes)|yellowing|gelbsucht|gelbe haut|augen (sind )?gelb|dark urine|dunkler urin", "yellow skin or eyes, or dark urine"),
)

FEVER_C = 38.5
LOW_SPO2 = 92


@dataclass(frozen=True)
class Triage:
    urgent: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def triage(prompt: str, wearable: dict[str, Any]) -> Triage:
    text = prompt.lower()
    reasons = [label for pattern, label in RED_FLAGS if re.search(pattern, text)]
    temperature, spo2 = wearable.get("temperature_c"), wearable.get("spo2")
    if isinstance(temperature, (int, float)) and temperature >= FEVER_C:
        reasons.append(f"a high temperature ({temperature:.1f} C) measured by your ring")
    if isinstance(spo2, (int, float)) and spo2 < LOW_SPO2:
        reasons.append(f"a low blood oxygen reading ({spo2:.0f} percent) from your ring")
    return Triage(bool(reasons), tuple(reasons))


def cautions(prompt: str) -> tuple[str, ...]:
    text = prompt.lower()
    return tuple(label for pattern, label in CAUTIONS if re.search(pattern, text))
