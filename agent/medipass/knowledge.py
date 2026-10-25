"""The rule set behind the traffic light. Small on purpose, so every line can be checked by a pharmacist.

DEMO DATA. It follows the direction of well-known drug label warnings and CPIC pharmacogenetic guidance
(clopidogrel with CYP2C19, codeine and tramadol with CYP2D6, fluoropyrimidines with DPYD), but it is
not a complete or validated database. Have a pharmacist review every rule before real use.
"""

from dataclasses import dataclass
from typing import Literal

Light = Literal["green", "yellow", "red"]

# Ingredient names (lowercase) and brand names or spellings that map to them.
ALIASES: dict[str, str] = {
    "ibuprofen": "ibuprofen",
    "nurofen": "ibuprofen",
    "clopidogrel": "clopidogrel",
    "plavix": "clopidogrel",
    "acetylsalicylic acid": "acetylsalicylic acid",
    "acetylsalicylsaeure": "acetylsalicylic acid",
    "acetylsalicylsäure": "acetylsalicylic acid",
    "aspirin": "acetylsalicylic acid",
    "ass": "acetylsalicylic acid",
    "omeprazole": "omeprazole",
    "omeprazol": "omeprazole",
    "warfarin": "warfarin",
    "amoxicillin": "amoxicillin",
    "codeine": "codeine",
    "codein": "codeine",
    "tramadol": "tramadol",
    "fluorouracil": "fluorouracil",
    "5-fu": "fluorouracil",
    "capecitabine": "capecitabine",
    "capecitabin": "capecitabine",
    "paracetamol": "paracetamol",
    "acetaminophen": "paracetamol",
    "pantoprazole": "pantoprazole",
    "pantoprazol": "pantoprazole",
    "simvastatin": "simvastatin",
    "ramipril": "ramipril",
}

# Medicines the demo data knows about. Anything else is reported as unknown, never as green.
KNOWN: frozenset[str] = frozenset(ALIASES.values())

CONDITION_ALIASES: dict[str, str] = {
    "stomach ulcer": "ulcer",
    "gastric ulcer": "ulcer",
    "magengeschwuer": "ulcer",
    "magengeschwür": "ulcer",
    "ulcer": "ulcer",
    "chronic kidney disease": "ckd",
    "ckd": "ckd",
    "niereninsuffizienz": "ckd",
}

ALLERGY_CLASSES: dict[str, frozenset[str]] = {
    "penicillin": frozenset({"amoxicillin"}),
    "nsaid": frozenset({"ibuprofen", "acetylsalicylic acid"}),
}

PHENOTYPE_ALIASES: dict[str, str] = {
    "poor": "poor",
    "pm": "poor",
    "intermediate": "intermediate",
    "im": "intermediate",
    "normal": "normal",
    "nm": "normal",
    "rapid": "rapid",
    "ultrarapid": "ultrarapid",
    "um": "ultrarapid",
    "decreased": "decreased",
    "decreased function": "decreased",
    "poor function": "poor",
}


@dataclass(frozen=True)
class Rule:
    light: Light
    title: str
    detail: str
    source: str
    advice: str = ""


# Two medicines taken together. Keys are unordered pairs.
INTERACTIONS: dict[frozenset[str], Rule] = {
    frozenset({"ibuprofen", "clopidogrel"}): Rule("yellow", "Higher bleeding risk", "Ibuprofen and clopidogrel each raise the risk of bleeding. Ask a pharmacist or doctor before combining them.", "Drug labels, NSAID with antiplatelet", advice="Paracetamol is usually the better choice for you. Ask a pharmacist."),
    frozenset({"ibuprofen", "acetylsalicylic acid"}): Rule("yellow", "May weaken the aspirin effect", "Ibuprofen can interfere with the heart-protecting effect of low-dose aspirin and adds to the bleeding risk.", "FDA information on ibuprofen with aspirin"),
    frozenset({"ibuprofen", "warfarin"}): Rule("red", "Serious bleeding risk", "Ibuprofen with a blood thinner such as warfarin can cause serious bleeding. Do not combine without medical advice.", "Drug labels, NSAID with anticoagulant", advice="Do not use ibuprofen. Ask a pharmacist for a safer painkiller."),
    frozenset({"acetylsalicylic acid", "clopidogrel"}): Rule("yellow", "Adds to the bleeding risk", "Aspirin on top of clopidogrel raises the bleeding risk. Some people are prescribed both on purpose, so do not add extra aspirin for a headache without asking.", "Drug labels, antiplatelet combinations"),
    frozenset({"acetylsalicylic acid", "warfarin"}): Rule("red", "Serious bleeding risk", "Aspirin with a blood thinner such as warfarin can cause serious bleeding. Do not combine without medical advice.", "Drug labels, aspirin with anticoagulant"),
    frozenset({"ibuprofen", "ramipril"}): Rule("yellow", "Can weaken blood pressure treatment and strain the kidneys", "Ibuprofen can reduce the effect of ACE inhibitors such as ramipril and, together, they can harm the kidneys.", "Drug labels, NSAID with ACE inhibitor", advice="Paracetamol is usually the better choice for you. Ask a pharmacist."),
    frozenset({"omeprazole", "clopidogrel"}): Rule("yellow", "Omeprazole can weaken clopidogrel", "Omeprazole reduces how well the body activates clopidogrel. Ask about an alternative stomach protector.", "Clopidogrel drug label"),
}

# A medicine against a gene result. Keys are (medicine, gene, phenotype).
PHARMACOGENETICS: dict[tuple[str, str, str], Rule] = {
    ("clopidogrel", "CYP2C19", "poor"): Rule("red", "Clopidogrel may not work for you", "People with this gene result activate clopidogrel poorly, so protection against clots is reduced. An alternative is usually recommended.", "CPIC, clopidogrel and CYP2C19", advice="Ask your cardiologist whether prasugrel or ticagrelor would suit you better."),
    ("clopidogrel", "CYP2C19", "intermediate"): Rule("yellow", "Clopidogrel may work less well", "This gene result can reduce how well clopidogrel is activated. Talk to the prescriber.", "CPIC, clopidogrel and CYP2C19"),
    ("codeine", "CYP2D6", "ultrarapid"): Rule("red", "Risk of overdose effects", "Ultrarapid CYP2D6 turns codeine into morphine too fast, with a risk of serious side effects. Avoid it.", "CPIC, codeine and CYP2D6"),
    ("codeine", "CYP2D6", "poor"): Rule("red", "Codeine will not relieve pain", "Poor CYP2D6 activity means codeine is hardly converted into its active form. Ask for another painkiller.", "CPIC, codeine and CYP2D6"),
    ("codeine", "CYP2D6", "intermediate"): Rule("yellow", "Codeine may work less well", "Reduced CYP2D6 activity can weaken the effect. Watch how it works and ask if it does not help.", "CPIC, codeine and CYP2D6"),
    ("tramadol", "CYP2D6", "ultrarapid"): Rule("red", "Risk of overdose effects", "Ultrarapid CYP2D6 can raise the active form of tramadol too much. Avoid it.", "CPIC, opioids and CYP2D6"),
    ("tramadol", "CYP2D6", "poor"): Rule("red", "Tramadol will relieve less pain", "Poor CYP2D6 activity weakens the effect. Ask for another painkiller.", "CPIC, opioids and CYP2D6"),
    ("fluorouracil", "DPYD", "poor"): Rule("red", "Risk of severe toxicity", "Little or no DPYD activity can cause severe or fatal side effects. Do not take it without a specialist's plan.", "CPIC, fluoropyrimidines and DPYD"),
    ("fluorouracil", "DPYD", "intermediate"): Rule("yellow", "Dose usually needs lowering", "Reduced DPYD activity raises the risk of side effects. The dose is normally reduced by a specialist.", "CPIC, fluoropyrimidines and DPYD"),
    ("capecitabine", "DPYD", "poor"): Rule("red", "Risk of severe toxicity", "Little or no DPYD activity can cause severe or fatal side effects. Do not take it without a specialist's plan.", "CPIC, fluoropyrimidines and DPYD"),
    ("capecitabine", "DPYD", "intermediate"): Rule("yellow", "Dose usually needs lowering", "Reduced DPYD activity raises the risk of side effects. The dose is normally reduced by a specialist.", "CPIC, fluoropyrimidines and DPYD"),
    ("simvastatin", "SLCO1B1", "decreased"): Rule("yellow", "Higher risk of muscle problems", "Decreased SLCO1B1 function raises simvastatin levels in the blood. A lower dose or another statin is usually advised.", "CPIC, simvastatin and SLCO1B1", advice="Ask about a lower dose or another statin."),
    ("simvastatin", "SLCO1B1", "poor"): Rule("yellow", "Much higher risk of muscle problems", "Poor SLCO1B1 function raises simvastatin levels strongly. A lower dose or another statin is usually advised.", "CPIC, simvastatin and SLCO1B1", advice="Ask about a lower dose or another statin."),
    ("warfarin", "CYP2C9", "intermediate"): Rule("yellow", "Dose usually needs lowering", "Reduced CYP2C9 activity slows warfarin breakdown, so bleeding risk rises. The dose is normally adjusted by the prescriber.", "CPIC, warfarin and CYP2C9"),
    ("warfarin", "CYP2C9", "poor"): Rule("yellow", "Dose usually needs lowering", "Poor CYP2C9 activity slows warfarin breakdown a lot, so bleeding risk rises. The dose is normally adjusted by the prescriber.", "CPIC, warfarin and CYP2C9"),
    ("ibuprofen", "CYP2C9", "poor"): Rule("yellow", "Stays in the body longer", "Poor CYP2C9 activity slows ibuprofen breakdown. Use the lowest dose and ask a pharmacist.", "CPIC, NSAIDs and CYP2C9"),
}

# A medicine against a known condition. Keys are (medicine, condition).
CONDITIONS: dict[tuple[str, str], Rule] = {
    ("ibuprofen", "ulcer"): Rule("red", "Can worsen a stomach ulcer", "Ibuprofen can irritate the stomach lining and cause bleeding in people with an ulcer.", "Drug labels, NSAID contraindications"),
    ("ibuprofen", "ckd"): Rule("yellow", "Can strain the kidneys", "Ibuprofen can reduce kidney function. Ask before using it with kidney disease.", "Drug labels, NSAID and kidney disease", advice="Paracetamol is usually the better choice for you. Ask a pharmacist."),
}

ALLERGY_RULE = Rule("red", "Matches a known allergy", "This medicine belongs to a group you are allergic to.", "Allergy record", advice="Do not take it. Ask your doctor or pharmacist for an alternative.")


@dataclass(frozen=True)
class SymptomLink:
    medicine: str
    gene: str
    phenotypes: frozenset[str]
    keywords: tuple[str, ...]
    message: str


# A complaint that may be explained by a medicine together with a gene result. Only a hint, never a diagnosis.
SYMPTOM_LINKS: tuple[SymptomLink, ...] = (
    SymptomLink("simvastatin", "SLCO1B1", frozenset({"decreased", "poor"}), ("muscle", "muskel", "myalg"), "Your muscle pain may be linked to simvastatin together with your SLCO1B1 result. Tell your doctor, and do not stop the medicine on your own."),
)
