"""Standalone embedding server using transformers (deployed on remote GPU server).

Replaces vLLM for embedding models because vLLM v0.19.1
doesn't properly serve /v1/embeddings for CausalLM-based embedders.
"""

import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/home/user/workspace/wangshuo/Models/Qwen3-embeddings/")
PORT = int(os.getenv("EMBED_PORT", "8002"))

app = FastAPI(title="Embedding Server")

tokenizer = None
model = None
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class EmbeddingRequest(BaseModel):
    model: str = "qwen3-embeddings"
    input: List[str]
    encoding_format: Optional[str] = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: dict


@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}, embedding server ready on port {PORT}")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-embeddings",
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    texts = request.input
    all_embeddings = []

    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use last hidden state, mean pooling over non-padding tokens
            attention_mask = inputs["attention_mask"]
            hidden = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask_expanded).sum(dim=1)
            counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.extend(embeddings.cpu().numpy().tolist())

    dim = len(all_embeddings[0]) if all_embeddings else 0
    data = [
        EmbeddingData(object="embedding", embedding=emb, index=j)
        for j, emb in enumerate(all_embeddings)
    ]

    return EmbeddingResponse(
        object="list",
        data=data,
        model=request.model,
        usage={"prompt_tokens": 0, "total_tokens": 0},
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)