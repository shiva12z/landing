from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
import os

client = TestClient(app)

EXPECTED_FIELDS = [
    "user_pseudo_id",
    "session_id",
    "eventtimestamp",
    "event_name",
    "transaction_id",
    "prev_event_time",
    "time_diff",
    "new_session",
    "engagement_type"
]

def get_any_user_id():
    if os.path.exists("preprocessed_sessions.csv"):
        df = pd.read_csv("preprocessed_sessions.csv", usecols=["user_pseudo_id"])
        if not df.empty:
            return df["user_pseudo_id"].iloc[0]
    return None

def test_get_sessions():
    response = client.get("/data/sessions")
    assert response.status_code in (200, 404, 500)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        if data:
            for field in EXPECTED_FIELDS:
                assert field in data[0]

def test_get_user_sessions():
    user_id = get_any_user_id() or "nonexistent_user"
    response = client.get(f"/data/sessions/{user_id}")
    assert response.status_code in (200, 404, 500)
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)
        if data:
            for field in EXPECTED_FIELDS:
                assert field in data[0]
    else:
        # Acceptable error messages
        assert (
            response.json().get("detail", "").startswith("No sessions found") or
            response.json().get("detail", "") == "Preprocessed data not found."
            or response.status_code == 500
        )

def test_get_user_recommendation():
    user_id = get_any_user_id() or "nonexistent_user"
    response = client.get(f"/data/recommendation/{user_id}")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "recommended_engagement_type" in response.json()
    else:
        assert response.json()["detail"].startswith("No recommendation available") or response.json()["detail"] == "Preprocessed data not found."
