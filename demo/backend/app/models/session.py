from pydantic import BaseModel
from typing import Optional

class Session(BaseModel):
    user_pseudo_id: str
    session_id: int
    eventtimestamp: Optional[str]
    event_name: Optional[str]
    transaction_id: Optional[str]
    prev_event_time: Optional[str]
    time_diff: Optional[float]
    new_session: Optional[bool]
    engagement_type: Optional[str]
    # Add more fields as needed to match your CSV output
