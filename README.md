# LocalLLM

LocalLLM ist ein Framework zum lokalen Betrieb und zur Anpassung von Large Language Models (LLMs). Damit lassen sich Modelle ohne externe Cloud-Services einsetzen und weiterentwickeln – ideal für datenschutzsensitive Szenarien.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Lokale Ausführung**: Keine Datenübertragung an externe Server notwendig.  
- **Einfache Integration**: Zugriff über Python-Klassen und (optional) über eine lokale REST-API.  
- **Anpassbarkeit**: Feintuning auf individuelle Anforderungen.  
- **Multi-Model-Unterstützung**: Kann mit verschiedenen Architekturen (z.B. GPT, LLaMA, Qwen) umgehen.  
- **Optimierungen**: Moderne Techniken zur Reduzierung von Rechenaufwand.

## Requirements

- **Python**: Version 3.9 oder höher (siehe `setup.py`)
- **RAM**: Mindestens 16 GB (empfohlen 32 GB)
- **GPU (optional)**: CUDA-fähig für beschleunigtes Inferencing

Benötigte Python-Bibliotheken (u.a.):

- `torch`
- `transformers`
- `numpy`
- `fastapi`
- `uvicorn`
- `tqdm`
- `openai` (nur falls API-Nutzung gewünscht)

Zusätzlich wird in `scripts/start.py` per `docker-compose` ein Docker-Setup aufgerufen. Dafür sollte **Docker** installiert sein, wenn man den Server innerhalb eines Containers laufen lassen möchte.

## Installation

1. **Repository klonen**:
   ```bash
   git clone https://github.com/Rob9999/LocalLLM.git
   cd LocalLLM
   ```

2. **Abhängigkeiten installieren** (Beispiel, falls eine `requirements.txt` vorhanden ist oder man selbst ein virtuelles Environment erstellt):
   ```bash
   pip install -r requirements.txt
   ```
   *Alternativ* manuell:
   ```bash
   pip install torch transformers numpy fastapi uvicorn tqdm openai
   ```

3. (Optional) **GPU konfigurieren**, wenn CUDA verwendet werden soll:
   - Stellen Sie sicher, dass `torch` mit CUDA-Unterstützung installiert ist.
   - Prüfen mit:
     ```bash
     python -c "import torch; print(torch.cuda.is_available())"
     ```

## Usage

### Lokalen Server starten

Zum Starten des lokalen LLM-Webservers gibt es ein Start-Skript:

```bash
python scripts/start.py
```

- Dieser Befehl ruft intern `docker-compose up -d` auf (sofern vorhanden) und startet anschließend eine FastAPI-Anwendung (Port 8000).  
- **Ohne Docker** könnte man alternativ direkt das Skript `local_LLM/webserver/webserver.py` per `uvicorn` ausführen:
  ```bash
  uvicorn local_LLM.webserver.webserver:app --host 0.0.0.0 --port 8000
  ```
  
Nach erfolgreichem Start ist der API-Endpunkt unter `http://localhost:8000` erreichbar.

### Model laden und benutzen (Python)

Falls Sie lieber nur lokal in Python Code ausführen möchten (ohne Webserver):

```python
from local_LLM.gpt_model_wrapper import GPTModelWrapper

# Initialisierung
wrapper = GPTModelWrapper(
    model_name="Qwen/Qwen2.5-7B-Instruct", 
    use_api=False, 
    start_model=True
)

# Text generieren
prompt = "Hello, how are you?"
output_text = wrapper.generate_text(prompt)
print(output_text)
```

## Training

Der LLM-Webserver enthält Endpunkte zum Trainieren. Beispiel via REST-API:

```bash
curl -X POST "http://localhost:8000/train/" \
  -H "Content-Type: application/json" \
  -d '{
    "training_data": ["Erster Trainingssatz", "Zweiter Trainingssatz"],
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 5e-5
  }'
```

- Über den Endpunkt `/training_status/` kann man den aktuellen Trainingsstatus abfragen.  
- Das Training läuft asynchron; das Modell wird automatisch in der projektspezifischen Ordnerstruktur gespeichert.

Für direktes Feintuning in Python (ohne API) bietet die Klasse `GPTModelWrapper` passende Methoden (`train()` oder `start_training_async()`).

## Contributing

Beiträge sind willkommen!  
- **Bug melden**: Bitte ein Issue auf GitHub erstellen.  
- **Neue Features**: Pull Request mit kurzer Beschreibung.  
- **Entwicklungsumgebung**: Falls eigene Ideen, gern einen Fork erstellen und dort arbeiten.

## License

Dieses Projekt steht unter der Mozilla Public License 2.0 (MPL-2.0). Siehe [LICENSE](LICENSE) für Details.
