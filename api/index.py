from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import json

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    messages: List[Any]
    schema: Dict[str, Any]
    temperature: Optional[float] = 0.7
    maxTokens: Optional[int] = None


class Query(BaseModel):
    queryName: str
    queryDescription: str
    sql: str


class GenerateResponse(BaseModel):
    queries: List[Query]


@app.post("/api/chat/completions")
async def generate(request: GenerateRequest) -> Dict[str, Any]:
    try:
        # Here you would typically call your LLM service
        # For now, we'll return a mock response that matches the schema

        # Extract the expected schema from the request
        schema = request.schema

        # Mock response - in production, replace with actual LLM call
        response = {
            "queries": [
                {
                    "queryName": "Sample Query",
                    "queryDescription": "This is a sample SQL query",
                    "sql": "SELECT * FROM example_table LIMIT 10",
                }
            ]
        }

        # Validate response against schema
        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
