# StyleZone Fashion E-Commerce Platform

A modern fashion e-commerce web application with a FastAPI backend and a React/Vite/Tailwind frontend.

## Getting Started

### 1. Clone the Repository

```
git clone <your-repo-url>
cd landing/demo
```

### 2. Set Up the Backend

#### a. Create a Virtual Environment (Recommended)

```
python -m venv .venv
```

#### b. Activate the Virtual Environment
- **Windows:**
  ```
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```
  source .venv/bin/activate
  ```

#### c. Install Backend Requirements

```
pip install -r backend/requirements.txt
```

#### d. Run the FastAPI Backend

```
cd backend
uvicorn app.main:app --reload
```

The backend will be available at `http://127.0.0.1:8000`.

### 3. Set Up the Frontend

```
cd ../frontend
npm install
npm run dev
```

The frontend will be available at the URL shown in your terminal (usually `http://localhost:5173`).

### 4. Usage
- Visit the frontend in your browser.
- Register a new account or log in with your credentials.
- Explore the shop and other features.

### 5. Project Structure

```
demo/
  backend/      # FastAPI backend (API, models, database)
  frontend/     # React + Vite + Tailwind frontend
```

### 6. Notes
- Make sure the backend is running before using the frontend login/register features.
- Update `.env` or config files as needed for production.
- Data and large files are ignored by git (see `.gitignore`).

### 7. License

MIT License
