# Texte für /submit (zum Kopieren)

Jeder Abschnitt ist ein Feld. Kopiere den Text darunter ins gleiche Feld auf https://superteam-agent-studio.vercel.app/submit. Entwürfe speichern dort automatisch.

Für die 50 Punkte "Idea locked in" reichen Titel und Problem im Build Track.

---

# BUILD TRACK

## Project title

MediPass. Does this medicine work for me? Medicines, genes and an AI agent in one app

## Problem solved

People take medicines that can clash with each other, with their allergies and with their genes, and nobody sees the whole picture. The electronic patient record is incomplete, for example it does not contain over-the-counter medicines such as ibuprofen. Gene results that predict how a person processes a drug (CYP2C19 with clopidogrel, CYP2D6 with codeine, DPYD with fluorouracil) sit in a separate report. Kidney values sit in a lab file. A patient on clopidogrel after a stent who buys ibuprofen, or whose body cannot activate clopidogrel at all, gets no warning at the moment of the decision. Doctors and pharmacists each see only part of the record. Patients, pharmacists and doctors are affected, and the cost is avoidable harm and treatment that quietly does not work.

## Project description

MediPass shows a traffic light (green, yellow, red) for a medicine against the person's own record, which holds current medicines, allergies, kidney values, gene results and notes on complaints. It also reviews every medicine the person already takes and gives a concrete next step, and it can point out when a complaint may be explained by a medicine together with a gene result. An AI care agent reads a free sentence such as "I have a very strong headache, get me tablets", checks for red flags first, compares the over-the-counter options with the record and the data from a smart ring, and orders the safest one. Prescription medicines wait for a quick yes or no from a doctor who sees the record and the safety check, with no appointment. The agent pays the doctor and the pharmacy with USDC on Solana devnet. A pharmacist can open a 15 minute code that shows warnings only and never the whole profile. MediPass does not replace advice from a doctor or pharmacist. The rules are a demo set and not a validated database, and the ring data is simulated.

## Code repository

(Hier kommt der GitHub-Link hinein, sobald das Repo existiert. Das Feld ist Pflicht.)

## Demo link (optional)

(Die öffentliche Adresse der App, sobald sie deployt ist.)

## Technical architecture

A Python and FastAPI agent service. A rule engine without any AI model holds drug interactions, gene and drug rules in the style of CPIC guidance, allergies and conditions, and returns the traffic light with a source for every finding. A care agent adds red flag triage, ranking of options, ordering and a doctor approval queue (Upstash Redis when deployed). A model layer (Claude, with Gemini as an alternative) only reads the sentence and writes the explanation. Payments use x402 and Solana devnet USDC with spending limits. The AI Race adapter speaks the ASRP/1 protocol for the verifier. Pharmacy codes are signed and expire. The web pages are plain HTML and JavaScript. More than 100 offline tests cover the rules, the agent, the payment logic and the protocol.

## AI capabilities

A language model (Claude) reads a free sentence in any language and extracts the complaint, a medicine the person names and possible emergency signs. It can only add caution. The traffic light always comes from fixed rules and never from the model, and everything the model returns is checked against fixed lists before use. The model also writes a short plain language explanation from the rule findings only. For the AI Race the model answers questions, extracts JSON that matches a schema, plans tool calls with retries and a lighter fallback model, answers from a source with citations (or null), and keeps conversation memory per session. We checked it with offline tests and live tests, including free sentences in German and French and emergency wording that our keyword rules missed but the model caught.

---

# POLICY TRACK (falls ihr beide Tracks abgebt)

## Proposal title

A consented health data pass. Safe medicine decisions without selling patient data

## Problem definition

Health data is scattered across the patient record, lab files, gene reports and the patient's own memory. Companies that collect wellness and gene data often sell it on, while the patients themselves get no safer medicine decisions from it. At the same time people take medicines that interact with their own genes, kidney function and other drugs, and AI agents are starting to buy things for us, including medicines. The people affected are patients, above all those with several medicines, and the clinicians and pharmacists who see only part of the picture.

## Proposed policy or governance framework

First, a patient owned health record, with the data kept on the person's device and shared only for a stated purpose and for a limited time. Second, a duty on every AI agent that orders medicines to check that record and to stop at red flags before it buys anything. Third, a human clinician in the loop for every prescription medicine, who sees the record and the safety check. Fourth, an audit trail for every decision, and short lived codes that show a pharmacist warnings only. Fifth, payments between agent, doctor and pharmacy that are logged and capped by spending limits.

## Research evidence

Gene and drug guidance from the Clinical Pharmacogenetics Implementation Consortium (CPIC) for clopidogrel with CYP2C19, codeine with CYP2D6 and fluoropyrimidines with DPYD, and drug label warnings for combinations such as ibuprofen with clopidogrel. In August 2026 Moderna and Merck reported a positive Phase 3 result for an individualised neoantigen therapy, which shows that medicine made for one patient is becoming real. Market reports on pharmacogenomics give a range of about 5.7 to 19.6 billion USD for 2025 depending on how the market is defined. Consumer health data firms such as CUDIS (a smart ring with data linked to Solana) and Nebula Genomics (sponsored sequencing with pharma partners) show models where the data is the product. Legal points such as the GDPR rules for health and genetic data, the German gene diagnostics act and access to the national patient record are our current understanding and still need a lawyer's check.

## Implementation considerations

The clinician who approves a prescription is responsible for the decision. The operator of the agent is responsible for its safety checks and its logs. The pharmacy is responsible for dispensing, and the patient for the accuracy of what they enter. Regulators would certify the rules, as for any medical software, and a first step is a pilot with a few pharmacies and doctors over the first year. Third party apps cannot simply read the national patient record, so people start by entering their medicines or scanning their medication plan. Genetic tests are ordered by doctors, so the app only takes in the result. Costs are small because a check is a software call and a payment of a few cents.

## Risks and limitations

Faster ordering must not mean lower quality of care, so red flags stop the order and a person always stays in the loop for prescriptions. Rules can be wrong or out of date, so every verdict names its source and says it does not replace a pharmacist. Rich health data can be misused, so we minimise what is shared and keep it on the device. Gene data can reveal facts about family members. Cheap and fast access can push people to skip a conversation with a professional. Access can be unequal if only some people have a ring or a gene test. Our rule set is a demo and not a validated database, and the ring data in our demo is simulated.
