"""What the agent can order. Demo catalog with demo prices in devnet USDC.

Prescription-only medicines never go straight to checkout. They wait for a doctor.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    ingredient: str
    label: str
    price_usdc: float
    prescription: bool
    treats: tuple[str, ...]


CATALOG: dict[str, Product] = {p.ingredient: p for p in (
    Product("paracetamol", "Paracetamol 500 mg, 20 tablets", 0.02, False, ("headache", "fever", "pain")),
    Product("ibuprofen", "Ibuprofen 400 mg, 20 tablets", 0.03, False, ("headache", "fever", "pain")),
    Product("acetylsalicylic acid", "Aspirin 500 mg, 20 tablets", 0.03, False, ("headache", "fever", "pain")),
    Product("pantoprazole", "Pantoprazole 20 mg, 14 tablets", 0.04, False, ("heartburn",)),
    Product("omeprazole", "Omeprazole 20 mg, 14 capsules", 0.04, False, ("heartburn",)),
    Product("amoxicillin", "Amoxicillin 500 mg, 20 capsules", 0.05, True, ()),
    Product("clopidogrel", "Clopidogrel 75 mg, 28 tablets", 0.05, True, ()),
    Product("simvastatin", "Simvastatin 20 mg, 28 tablets", 0.05, True, ()),
)}

INTENTS: tuple[tuple[str, str], ...] = (
    ("headache", r"headache|migraine|kopfschmerz|kopfweh|migräne|migraene|head (hurts|aches|is pounding|is throbbing)|pounding head|(schädel|schaedel|kopf) (brummt|dröhnt|droehnt|hämmert|haemmert|pocht|tut weh)|brummschädel|brummschaedel"),
    ("fever", r"fever|temperature|fieber"),
    ("heartburn", r"heartburn|acid reflux|sodbrennen|reflux|stomach acid|magensäure"),
    ("pain", r"\bpain\b|\bache\b|schmerz"),
)

CONSULT_FEE_USDC = 0.02
