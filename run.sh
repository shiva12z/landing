#!/bin/bash

# Exit on error
set -e

# 1. Setup Python virtual environment for backend
echo "[1/4] Setting up Python virtual environment for backend..."
cd demo/backend

# Check if Python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment"
    exit 1
  fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  source .venv/Scripts/activate
else
  echo "Error: Virtual environment activation script not found"
  exit 1
fi

# 2. Install backend dependencies
echo "[2/4] Installing backend dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Start backend server (in background)
echo "[3/4] Starting backend server..."
nohup python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# 4. Start frontend server (in background)
echo "[4/4] Starting frontend server..."
cd demo/frontend
npm install
nohup npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!
cd ../..

echo "---"
echo "Backend running (PID: $BACKEND_PID, log: demo/backend/backend.log)"
echo "Frontend running (PID: $FRONTEND_PID, log: demo/frontend/frontend.log)"
echo "---"
echo "Access the backend API at: http://localhost:8000/docs"
echo "Access the frontend at:   http://localhost:5173/"

echo "To stop the servers, run:"
echo "  kill $BACKEND_PID $FRONTEND_PID" 