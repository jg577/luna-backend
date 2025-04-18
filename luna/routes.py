from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from .generator import get_luna_generator

# Create API router for Luna endpoints
router = APIRouter()


class ChartRequest(BaseModel):
    """Request model for chart generation endpoint."""

    data: List[Dict[str, Any]]
    prompt: str
    options: Optional[Dict[str, Any]] = None


class SQLRequest(BaseModel):
    """Request model for SQL generation endpoint."""

    prompt: str
    schema: Dict[str, Any]
    prompt_template: Optional[str] = "prompt1"


@router.post("/api/charts/generate")
async def generate_chart(request: ChartRequest) -> Dict[str, Any]:
    """
    Generate chart configuration based on data and a natural language prompt.

    This endpoint uses the LunaGenerator to create appropriate chart configurations
    based on the provided data and the user's description of what visualization they want.
    """
    try:
        # Get Luna generator
        luna_generator = get_luna_generator()

        # Generate chart configuration
        response = await luna_generator.generate_chart(request.data, request.prompt)

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


@router.post("/api/sql/generate")
async def generate_sql(request: SQLRequest) -> Dict[str, Any]:
    """
    Generate SQL query based on a natural language prompt and schema.

    This endpoint uses the LunaGenerator to create appropriate SQL queries
    based on the provided schema and the user's natural language question.
    """
    try:
        # Get Luna generator
        luna_generator = get_luna_generator()

        # Generate SQL query using specified template
        response = await luna_generator.generate_sql_query(
            request.prompt, request.schema, prompt_template=request.prompt_template
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL: {str(e)}")


@router.get("/api/prompts")
async def list_prompts() -> Dict[str, List[str]]:
    """
    Get a list of available prompt templates.

    Returns the names of all available prompt templates that can be used.
    """
    try:
        # Get Luna generator
        luna_generator = get_luna_generator()

        # Return available prompt templates
        prompt_names = list(luna_generator.prompt_options.keys())
        return {"prompts": prompt_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing prompts: {str(e)}")
