from __future__ import annotations

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.evaluator.evaluate import QwenVLMJudge


app = FastAPI(title="Text Alignment Judge API")
judge: QwenVLMJudge | None = None


class EvalRequest(BaseModel):
    image_a_path: str
    image_b_path: str
    prompt_text: str
    checklist: list[str]


@app.on_event("startup")
def startup_event() -> None:
    global judge
    print("[AlignmentServer] Loading the alignment judge model...")
    judge = QwenVLMJudge()
    judge.load_model()
    print("[AlignmentServer] Alignment judge is ready.")


@app.post("/evaluate")
def evaluate(request: EvalRequest):
    try:
        return judge.evaluate_pair(
            image_a_path=request.image_a_path,
            image_b_path=request.image_b_path,
            prompt_text=request.prompt_text,
            checklist=request.checklist,
        )
    except Exception as exc:
        print(f"[AlignmentServer] Evaluation error: {exc}")
        return JSONResponse(status_code=422, content={"error": str(exc), "winner": "Tie"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
