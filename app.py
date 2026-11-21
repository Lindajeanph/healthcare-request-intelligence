"""Local-only research demonstration; never submit patient data."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from common import load_model, predict, parse_result

ROUTES = {
    "records_request": "route_to_roi_specialist",
    "billing": "route_to_billing",
    "insurance": "route_to_insurance",
    "appointment": "route_to_scheduling",
    "technical_support": "route_to_it",
}
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    adapter_path = os.environ.get("HRI_ADAPTER_PATH", "outputs/hri-lora")
    state["tokenizer"], state["model"] = load_model(adapter_path)
    try:
        yield
    finally:
        state.clear()

app = FastAPI(
    title="Healthcare Request Intelligence (HRI)",
    version="1.0.0",
    description="Demonstration only. Synthetic administrative text; human review required.",
    lifespan=lifespan,
)

class TriageRequest(BaseModel):
    text: str = Field(min_length=8, max_length=1500, description="Fictional administrative request only")

@app.get("/health")
def health():
    return {"status": "ready", "model": "Qwen3-0.6B + HRI LoRA", "research_only": True}

@app.post("/triage")
def triage(item: TriageRequest):
    try:
        raw = predict(state["tokenizer"], state["model"], item.text)
        result = parse_result(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Model inference failed") from exc
    if not result:
        return {"status": "manual_review", "result": None}
    if (result.get("category") not in ROUTES or
        result.get("priority") not in {"normal", "urgent"} or
        result.get("action") != ROUTES.get(result.get("category"))):
        return {"status": "manual_review", "result": None}
    return {"status": "model_prediction_requires_review", "result": result}
