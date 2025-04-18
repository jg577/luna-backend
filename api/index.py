from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Luna package
import luna
from luna.routes import router as luna_routes

# Set up OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    # In production, we don't prompt for input - we fail fast
    if not os.environ.get("VERCEL"):  # Check if running on Vercel
        import getpass

        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
    # If on Vercel without API key, the app will throw errors when using OpenAI

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Luna routes
app.include_router(luna_routes)


class GenerateRequest(BaseModel):
    messages: List[Any]
    schema: Dict[str, Any]
    temperature: Optional[float] = 0.7
    maxTokens: Optional[int] = None
    prompt_template: Optional[str] = "prompt1"


class Query(BaseModel):
    queryName: str
    queryDescription: str
    sql: str


class GenerateResponse(BaseModel):
    queries: List[Query]


@app.post("/api/chat/completions")
async def generate(request: GenerateRequest) -> Dict[str, Any]:
    try:
        # Log request info (without sensitive data)
        logger.info(f"Processing request with {len(request.messages)} messages")

        # Extract the user's prompt from the messages
        user_messages = [msg for msg in request.messages if msg.get("role") == "user"]

        if not user_messages:
            logger.warning("No user message found in the request")
            raise HTTPException(
                status_code=400, detail="No user message found in the request"
            )

        # Get the latest user message
        prompt = user_messages[-1].get("content", "")

        # Validate schema
        if not request.schema:
            logger.warning("Empty schema provided in request")
            raise HTTPException(status_code=400, detail="Schema is required")

        # Get Luna generator
        try:
            luna_generator = luna.get_luna_generator()
        except Exception as e:
            logger.error(f"Failed to initialize generator: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error initializing generator: {str(e)}"
            )

        # Generate SQL query with specified prompt template
        try:
            response = await luna_generator.generate_sql_query(
                user_input=prompt,
                schema=request.schema,
                prompt_template=request.prompt_template,
            )
            return response
        except Exception as e:
            logger.error(f"Error in SQL query generation: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500, detail=f"Error generating query: {str(e)}"
            )

    except HTTPException:
        # Re-raise HTTP exceptions without wrapping
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error generating response: {str(e)}"
        )


@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
