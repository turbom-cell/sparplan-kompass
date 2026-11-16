# Sparplan-Kompass

**Live-App:** https://sparplan-kompass.streamlit.app

Web-App für Privatanleger mit ETF-Sparplan: Depot-Risiko senken, **ohne zu verkaufen**.
Die App berechnet, wie die monatliche Sparrate aufgeteilt werden muss, damit das Depot
eine selbst gewählte Schmerzschwelle in Euro einhält, und zeigt daneben, was sofortiges
Umschichten an Steuern und Gebühren kosten würde.

## Funktionen

- Depot eingeben (Bestand und Einstand je ETF), Kauf-Kandidaten mit 0 € Bestand ergänzen
- Schmerzschwelle in Euro statt Risiko in Prozent (95 %-CVaR eines Börsentags)
- Ziel-Depot mit der höchsten erwarteten Rendite innerhalb der Schmerzschwelle, allein durch Zukäufe erreichbar (Nebenbedingung „nichts verkaufen“)
- Kurshistorie ab 2000: vor Auflage der ETFs stehen Ersatz-Reihen (Index oder älterer Fonds in Euro), in der App offen ausgewiesen
- Feste Sparplan-Aufteilung für den gewählten Zeitraum, animierter Weg ins Ziel
- Steuervergleich: sofort umschichten gegen Sparplan (Abgeltungsteuer, Teilfreistellung, Pauschbetrag)
- Marktvergleich mit echten Kursen: Sparplan-Kompass gegen „weiter wie bisher“ und einen Markt-ETF, ohne Blick in die Zukunft
- Stresstest in historischen Krisen und Blindtest auf ungesehenen Daten

## Aufbau

| Datei | Aufgabe |
|---|---|
| `app.py` | Oberfläche (Streamlit): Eingaben, Tabs, Texte |
| `ui/charts.py` | Grafiken (Plotly) |
| `ui/theme.py` | Design: Farben, CSS, Kennzahl-Karten |
| `core/data_service.py` | Kurse laden (Yahoo Finance), Datenfehler filtern, Ersatz-Reihen ansetzen, Notfall-Snapshot |
| `core/optimizer.py` | CVaR, sicherstes Depot und Ziel-Depot (lineare Programme, HiGHS) |
| `core/rebalancer.py` | Sparplan-Fahrplan ohne Verkäufe |
| `core/tax.py` | Steuer und Gebühren beim Verkauf |
| `core/analytics.py` | Marktvergleich (Walk-Forward), Krisen-Stresstest, Blindtest |
| `tests/test_core.py` | Tests für den Rechenkern |
| `data/sample_prices.csv` | Kurs-Snapshot, falls Yahoo Finance nicht antwortet |
| `data/proxy_history.csv` | Ersatz-Reihen in Euro für die Zeit vor ETF-Auflage (ab 1999) |

`core/` kennt weder Streamlit noch Plotly und lässt sich deshalb ohne Oberfläche testen.

## Lokal starten

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

Tests:

```bash
pip install pytest
```

```bash
python -m pytest
```

Kurs-Snapshot erneuern (vor der Präsentation sinnvoll):

```bash
python scripts/update_sample.py
```

## Deployment (Streamlit Community Cloud, kostenlos)

1. Diesen Ordner als eigenes GitHub-Repository hochladen.
2. Auf share.streamlit.io mit dem GitHub-Konto anmelden, „Create app“ wählen.
3. Repository, Branch `main` und Hauptdatei `app.py` auswählen, „Deploy“.
4. Jeder Push auf `main` aktualisiert die Live-App automatisch.

Antwortet Yahoo Finance in der Cloud nicht, nutzt die App automatisch den Snapshot
und zeigt das oben in der App an.

## Grenzen

- Historische Kurse sagen die Zukunft nicht voraus.
- Kursbewegungen während der Sparplan-Laufzeit werden nicht prognostiziert.
- Vereinfachte Steuerrechnung (ohne Kirchensteuer, Vorabpauschale, FIFO). Keine Anlage- oder Steuerberatung.
