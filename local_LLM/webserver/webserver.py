from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
from typing import List
from local_LLM.gpt_model_wrapper import GPTModelWrapper, TrainingStatus
from local_LLM.protocol import Protocol


# Pydantic-Modelle für die API-Eingaben
class TextGenerationRequest(BaseModel):
    input_text: str
    max_length: int = 50
    max_new_tokens: int = None


class TrainingRequest(BaseModel):
    training_data: List[str]
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 5e-5


class LLMWebServer:
    def __init__(self, model: GPTModelWrapper = None):
        # Logger/Protokollierung
        self.protocol = Protocol()

        # FastAPI-App erstellen
        self.app = FastAPI()

        # GPTModelWrapper initialisieren oder übernehmen
        if not model:
            self.model = GPTModelWrapper()
        else:
            self.model = model

        # Routen definieren (Endpunkte)
        @self.app.post("/generate_text/")
        def _generate_text(request: TextGenerationRequest):
            return self.generate_text(request)

        @self.app.post("/train/")
        def _train_model(request: TrainingRequest):
            return self.train_model(request)

        @self.app.get("/training_status/")
        def _training_status():
            return self.get_training_status()

        @self.app.post("/stop_model/")
        def _stop_model():
            return self.stop_model()

        @self.app.post("/restart_model/")
        def _restart_model():
            return self.restart_model()

    def start_web_server(self):
        import uvicorn

        self.protocol.info("Starte den Webserver ...")
        uvicorn.run(self.app, host="0.0.0.0", port=8000)
        self.protocol.info("Webserver wurde gestartet.")

    # --- Methoden für die Endpunkte ---

    def generate_text(self, request: TextGenerationRequest):
        # Generierungs-Parameter setzen
        self.model.set_generation_parameters(
            max_length=request.max_length,
            max_new_tokens=request.max_new_tokens,
        )
        # Text generieren
        output_text = self.model.generate_text(request.input_text)
        return {"generated_text": output_text}

    def train_model(self, request: TrainingRequest):
        # Prüfen, ob gerade ein Training läuft
        if self.model.get_training_status() in [
            TrainingStatus.STARTED,
            TrainingStatus.PENDING,
        ]:
            raise HTTPException(status_code=400, detail="Training läuft bereits.")

        # Training asynchron in separatem Thread starten
        threading.Thread(
            target=self.model.start_training_async,
            args=(
                request.training_data,
                request.epochs,
                request.batch_size,
                request.learning_rate,
            ),
        ).start()
        return {"status": "Training gestartet"}

    def get_training_status(self):
        status = self.model.get_training_status()
        return {"training_status": status.name}

    def stop_model(self):
        self.model.stop()
        return {"status": "Model gestoppt"}

    def restart_model(self):
        self.model.restart()
        return {"status": "Model neu gestartet"}


# Zum direkten Starten über die Kommandozeile
if __name__ == "__main__":
    server = LLMWebServer()
    server.start_web_server()
