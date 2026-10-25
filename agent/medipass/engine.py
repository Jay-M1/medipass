"""Checks one medicine against a profile and returns a traffic light with the reasons.

Pure functions, no network and no AI model. The light is the most severe finding. A medicine the rules
do not know is reported as "unknown", never as green, because silence is not safety.
"""

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .knowledge import (
    ALIASES,
    ALLERGY_CLASSES,
    ALLERGY_RULE,
    CONDITION_ALIASES,
    CONDITIONS,
    INTERACTIONS,
    KNOWN,
    PHARMACOGENETICS,
    PHENOTYPE_ALIASES,
    SYMPTOM_LINKS,
    Light,
)

Verdict = Literal["green", "yellow", "red", "unknown"]
SEVERITY: dict[str, int] = {"green": 0, "yellow": 1, "red": 2}

EGFR_REDUCED = 60  # below this the kidneys work less well

DISCLAIMER = "MediPass does not replace advice from a doctor or pharmacist. Demo rule set, not a validated database."


@dataclass(frozen=True)
class Profile:
    medications: tuple[str, ...] = ()
    allergies: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()
    genotype: dict[str, str] = field(default_factory=dict)
    labs: dict[str, float] = field(default_factory=dict)
    symptoms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Finding:
    light: Light
    kind: Literal["interaction", "pharmacogenetics", "condition", "allergy"]
    title: str
    detail: str
    source: str
    advice: str = ""


@dataclass(frozen=True)
class CheckResult:
    verdict: Verdict
    medicine: str | None
    findings: tuple[Finding, ...]
    notes: tuple[str, ...]
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalise_medicine(name: str) -> str | None:
    """Maps a brand name or spelling to its ingredient, or None if the rules do not know it."""
    return ALIASES.get(name.strip().lower())


def parse_profile(raw: dict[str, Any]) -> Profile:
    def strings(key: str) -> tuple[str, ...]:
        value = raw.get(key) or []
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f'"{key}" must be a list of strings.')
        return tuple(value)

    genotype = raw.get("genotype") or {}
    if not isinstance(genotype, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in genotype.items()):
        raise ValueError('"genotype" must map gene names to phenotypes, for example {"CYP2C19": "poor"}.')
    labs = raw.get("labs") or {}
    if not isinstance(labs, dict) or not all(isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool) for k, v in labs.items()):
        raise ValueError('"labs" must map names to numbers, for example {"egfr": 52}.')
    return Profile(strings("medications"), strings("allergies"), strings("conditions"), dict(genotype), {k.lower(): float(v) for k, v in labs.items()}, strings("symptoms"))


def check(medicine: str, profile: Profile) -> CheckResult:
    ingredient = normalise_medicine(medicine)
    if ingredient is None:
        return CheckResult("unknown", None, (), (f'"{medicine}" is not in the demo rule set, so no verdict is given.',))

    findings: list[Finding] = []
    notes: list[str] = []

    for taken in profile.medications:
        other = normalise_medicine(taken)
        if other is None:
            notes.append(f'Current medicine "{taken}" is not in the demo rule set and was not checked.')
            continue
        rule = INTERACTIONS.get(frozenset({ingredient, other}))
        if rule and other != ingredient:
            findings.append(Finding(rule.light, "interaction", rule.title, f"With {other}. {rule.detail}", rule.source, rule.advice))

    for gene, phenotype in profile.genotype.items():
        normalised = PHENOTYPE_ALIASES.get(phenotype.strip().lower())
        if normalised is None:
            notes.append(f'Gene result "{phenotype}" for {gene} was not understood and was not checked.')
            continue
        rule = PHARMACOGENETICS.get((ingredient, gene.strip().upper(), normalised))
        if rule:
            findings.append(Finding(rule.light, "pharmacogenetics", rule.title, f"{gene.upper()} {normalised} {'function' if gene.strip().upper() == 'SLCO1B1' else 'metabolizer'}. {rule.detail}", rule.source, rule.advice))

    conditions = [(c, CONDITION_ALIASES.get(c.strip().lower())) for c in profile.conditions]
    egfr = profile.labs.get("egfr")
    if egfr is not None and egfr < EGFR_REDUCED:
        conditions.append((f"reduced kidney function (eGFR {egfr:g})", "ckd"))
    for condition, key in conditions:
        rule = CONDITIONS.get((ingredient, key)) if key else None
        if rule:
            findings.append(Finding(rule.light, "condition", rule.title, f"With {condition}. {rule.detail}", rule.source, rule.advice))

    for allergy in profile.allergies:
        if ingredient in ALLERGY_CLASSES.get(allergy.strip().lower(), frozenset()):
            findings.append(Finding(ALLERGY_RULE.light, "allergy", ALLERGY_RULE.title, f"Allergy to {allergy}. {ALLERGY_RULE.detail}", ALLERGY_RULE.source, ALLERGY_RULE.advice))

    verdict: Verdict = max((f.light for f in findings), key=SEVERITY.__getitem__, default="green")
    return CheckResult(verdict, ingredient, tuple(sorted(findings, key=lambda f: -SEVERITY[f.light])), tuple(notes))


def known_medicines() -> list[str]:
    return sorted(KNOWN)


def review(profile: Profile) -> dict[str, Any]:
    """Checks every medicine the person already takes against the rest of their record, worst first."""
    items: list[dict[str, Any]] = []
    for taken in profile.medications:
        ingredient = normalise_medicine(taken)
        if ingredient is None:
            items.append({"medicine": taken, "verdict": "unknown", "findings": [], "mayExplain": []})
            continue
        others = tuple(m for m in profile.medications if normalise_medicine(m) != ingredient)
        result = check(taken, replace(profile, medications=others))
        phenotypes = {gene.strip().upper(): PHENOTYPE_ALIASES.get(value.strip().lower()) for gene, value in profile.genotype.items()}
        complaints = " ".join(profile.symptoms).lower()
        links = [link.message for link in SYMPTOM_LINKS if link.medicine == ingredient and phenotypes.get(link.gene) in link.phenotypes and any(k in complaints for k in link.keywords)]
        items.append({"medicine": ingredient, "verdict": result.verdict, "findings": [asdict(f) for f in result.findings], "mayExplain": links})
    items.sort(key=lambda i: -SEVERITY.get(i["verdict"], 0.5))
    return {"items": items, "disclaimer": DISCLAIMER}
