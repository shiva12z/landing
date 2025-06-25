# Backend for Hyper-Personalized Landing Page Generator

This backend provides data preprocessing, API endpoints, and a foundation for ML-driven personalization.

## Structure
- `preprocess.py`: Data preprocessing script
- `app/`: FastAPI application
  - `routes/`: API route definitions
  - `models/`: Pydantic models and data schemas
  - `services/`: Business logic and ML integration
- `config/`: Configuration files
- `tests/`: Test cases

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run preprocessing: `python preprocess.py`
3. Start API: `uvicorn app.main:app --reload`
