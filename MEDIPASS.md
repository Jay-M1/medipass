# MediPass, Stand für die Abgabe

MediPass zeigt per Ampel, ob ein Medikament bei dir persönlich passt. Es prüft gegen deine Medikamente, Allergien, Nierenwerte und Gen-Ergebnisse. Ein Agent versteht freie Sätze, bestellt sichere rezeptfreie Mittel und holt bei Rezepten ein schnelles Ja oder Nein von einem Arzt. Die Zahlungen laufen als Agent-Zahlungen auf Solana Devnet.

## Seiten

| Seite | Inhalt |
|---|---|
| `/app` | Ampel-Check, Medikations-Überprüfung, Profil von Anna, Apotheken-Code, Notfallpass |
| `/agent` | Der Agent. Prompt eingeben, er prüft, bestellt oder fragt den Arzt |
| `/doctor` | Der Arzt genehmigt oder lehnt ab (Demo ohne Login) |
| `/vision` | Der Pitch mit Plan, Markt, Wettbewerbern und Policy-Vorschlag |
| `/r/<code>` | Was die Apotheke sieht, nur Warnhinweise |

## Starten

```
cd medipass
source .venv/bin/activate
python -m agent serve
```

Dann `http://localhost:3000/app` öffnen. Die Tests laufen offline mit `python -m unittest discover -s tests`.

## Schlüssel in `.env`

| Variable | Wofür |
|---|---|
| `ANTHROPIC_API_KEY` oder `GEMINI_API_KEY` | Das Modell. Mit Claude-Schlüssel nutzt die App Claude, sonst Gemini. `LLM_PROVIDER` erzwingt eine Wahl |
| `AGENT_STUDIO_VERIFIER_TOKEN` | Von der Seite `/team`, für die Race-Prüfung |
| `AGENT_PRIVATE_KEY` | Wallet für die Zahlungsstufe (`python -m agent keygen`), mit Test-USDC und etwas Devnet-SOL |
| `MEDIPASS_PHARMACY_ADDRESS`, `MEDIPASS_DOCTOR_ADDRESS` | Zwei Devnet-Adressen. Ohne sie zeigt die App die Zahlung als simuliert |
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | Nur auf Vercel nötig. Hält die Arzt-Warteschlange und den Gesprächsverlauf zusammen |

## Deployment auf Vercel

1. Im Ordner `medipass` mit `npx vercel` anmelden und verknüpfen.
2. Die Variablen aus der Tabelle im Vercel-Dashboard unter Settings und Environment Variables eintragen. Die `.env` wird bewusst nicht hochgeladen.
3. Mit `npx vercel --prod` veröffentlichen. Die öffentliche Adresse kommt auf `/team` und dann auf "Test connection".
4. Auf `/race` die Stufen der Reihe nach verifizieren. Zuerst "App live", dann "AI Integration", "Agent uses tools", zuletzt die Zahlung.

Falls der Build auf Vercel scheitert, geht es mit dem mitgelieferten `Dockerfile` auf Railway oder Render. Dort braucht ihr kein Redis.

## Pitch in einer Minute

1. `/app`, Anna, 58. "Review my medicines" zeigt Clopidogrel in Rot (wirkt bei ihr kaum), Simvastatin in Gelb mit dem Hinweis auf ihre Muskelschmerzen.
2. "Ibuprofen" ergibt Gelb, mit drei Gründen und dem Rat zu Paracetamol. "Pantoprazole" ist Grün, "Amoxicillin" Rot.
3. `/agent`, "Ich habe sehr starke Kopfschmerzen, hol mir bitte Tabletten". Der Agent bestellt Paracetamol und zeigt jeden Schritt.
4. "Please order Simvastatin for me". Das geht an `/doctor`, der Arzt sieht die Gen-Warnung und genehmigt. Danach zahlt der Agent Arzt und Apotheke.
5. "This is the worst headache of my life". Der Agent bestellt nichts und verweist auf den Notruf.
6. `/vision` für Plan und Policy.

## Was ihr ehrlich sagen müsst

- Die Regeln sind ein Demo-Datensatz nach öffentlichen Leitlinien und nicht validiert.
- Die Ringdaten sind simuliert. Bestellungen sind Demo, Zahlungen laufen mit Test-Geld auf Devnet.
- Marktzahlen sind Spannen verschiedener Anbieter. Genomes.io konnte nicht belegt werden. Die Rechtsaussagen sind nicht frisch geprüft.
- Das Modell versteht nur den Satz. Es darf Vorsicht hinzufügen, aber nie eine Warnung der Regeln aufheben.
- MediPass ersetzt keine Beratung durch Arzt oder Apotheke.
