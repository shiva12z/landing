from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any
from backend.personalization_logic import PersonalizationEngine
import traceback

router = APIRouter()

# Initialize the personalization engine (load trends if needed)
personalization_engine = PersonalizationEngine()

@router.post("/personalize")
async def personalize(request: Request):
    """
    Accepts user/session data as JSON and returns personalized landing page content.
    """
    try:
        user_data: Dict[str, Any] = await request.json()
        user_profile = personalization_engine.create_user_profile(user_data)
        landing_page = personalization_engine.get_full_landing_page_content(user_profile)
        return JSONResponse(content=landing_page)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Personalization error: {str(e)}") 