# StyleZone - Hyper-Personalized Fashion E-Commerce Platform

A modern, AI-powered fashion e-commerce platform featuring hyper-personalized landing pages, intelligent product recommendations, and a seamless shopping experience. Built with FastAPI backend and React/Vite frontend.

## 🚀 Features

- **Hyper-Personalized Landing Pages**: AI-driven content personalization based on user behavior
- **Intelligent Product Recommendations**: ML-powered recommendation engine with cold-start strategies
- **User Segmentation**: Advanced user clustering and behavioral analysis
- **Real-time Personalization**: Dynamic content adaptation based on user interactions
- **Modern E-commerce UI**: Responsive design with Tailwind CSS
- **Secure Authentication**: User registration and login with password hashing
- **Session Analytics**: Comprehensive user session tracking and analysis

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (for backend)
- **Node.js 16+** (for frontend)
- **Git** (for cloning the repository)
- **pip** (Python package manager)
- **npm** (Node.js package manager)

### System Requirements

- **RAM**: Minimum 4GB (8GB recommended for ML operations)
- **Storage**: At least 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

## 🛠️ Installation & Setup

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone <your-repository-url>
cd landing/demo

# Verify the project structure
ls -la
```

**Expected Output:**
```
backend/          # FastAPI backend
frontend/         # React frontend
README.md         # This file
```

### Step 2: Backend Setup

#### 2.1 Navigate to Backend Directory
```bash
cd backend
```

#### 2.2 Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

**✅ Success Indicator:** You should see `(.venv)` at the beginning of your command prompt.

#### 2.3 Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Expected Output:**
```
Collecting fastapi...
Collecting uvicorn...
Collecting pandas...
...
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 pandas-2.1.3 ...
```

#### 2.4 Verify Backend Installation
```bash
# Check if all packages are installed
pip list

# Test Python imports
python -c "import fastapi, uvicorn, pandas, numpy, sqlalchemy; print('✅ All packages imported successfully')"
```

### Step 3: Frontend Setup

#### 3.1 Navigate to Frontend Directory
```bash
cd ../frontend
```

#### 3.2 Install Node.js Dependencies
```bash
npm install
```

**Expected Output:**
```
added 1234 packages, and audited 1235 packages in 1m
found 0 vulnerabilities
```

#### 3.3 Verify Frontend Installation
```bash
# Check if all packages are installed
npm list --depth=0

# Test if Vite is working
npm run build
```

## 🚀 Running the Application

### Step 1: Start the Backend Server

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment (if not already activated)
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Start the FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**✅ Success Indicators:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [1234] using WatchFiles
INFO:     Started server process [5678]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**🔗 Backend URLs:**
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

### Step 2: Start the Frontend Server

Open a **new terminal window** and run:

```bash
# Navigate to frontend directory
cd frontend

# Start the development server
npm run dev
```

**✅ Success Indicators:**
```
  VITE v6.3.5  ready in 234 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

**🔗 Frontend URL:** http://localhost:5173/

## 🐛 Troubleshooting & Error Handling

### Common Backend Issues

#### Issue 1: Port Already in Use
**Error:** `OSError: [Errno 98] Address already in use`

**Solution:**
```bash
# Find process using port 8000
netstat -tulpn | grep :8000  # Linux/macOS
netstat -an | findstr :8000  # Windows

# Kill the process
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows

# Or use a different port
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

#### Issue 2: Virtual Environment Not Activated
**Error:** `ModuleNotFoundError: No module named 'fastapi'`

**Solution:**
```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Verify activation
which python  # Should show path to .venv/bin/python
pip list  # Should show installed packages
```

#### Issue 3: Database Connection Error
**Error:** `sqlite3.OperationalError: unable to open database file`

**Solution:**
```bash
# Check file permissions
ls -la sessions.db

# Create database directory if needed
mkdir -p data
chmod 755 data
```

#### Issue 4: Missing Dependencies
**Error:** `ImportError: No module named 'pandas'`

**Solution:**
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Or install individually
pip install fastapi uvicorn pandas numpy sqlalchemy passlib[bcrypt] scikit-learn
```

### Common Frontend Issues

#### Issue 1: Port Already in Use
**Error:** `Port 5173 is already in use`

**Solution:**
```bash
# Use different port
npm run dev -- --port 3000

# Or kill existing process
npx kill-port 5173
```

#### Issue 2: Node Modules Corrupted
**Error:** `Cannot find module 'react'`

**Solution:**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### Issue 3: Build Errors
**Error:** `Build failed with errors`

**Solution:**
```bash
# Check for linting errors
npm run lint

# Fix auto-fixable issues
npm run lint -- --fix

# Clear build cache
rm -rf dist
npm run build
```

### Performance Issues

#### Issue 1: Slow Backend Startup
**Solution:**
```bash
# Disable reload for production
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Or use production server
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

#### Issue 2: Memory Issues with Large Datasets
**Solution:**
```bash
# Increase Python memory limit
export PYTHONMALLOC=malloc
python -m uvicorn app.main:app --reload

# Or use memory-efficient settings
python -X dev -m uvicorn app.main:app --reload
```

## 📊 API Endpoints

### Authentication Endpoints
- `POST /data/login` - User login
- `POST /data/register` - User registration

### Data Endpoints
- `GET /data/sessions` - Get all sessions
- `GET /data/sessions/{user_pseudo_id}` - Get user sessions
- `GET /data/sessions_db` - Get database sessions

### Personalization Endpoints
- `POST /personalize` - Get personalized content
- `GET /data/personalize/{user_pseudo_id}` - Personalize for known user
- `POST /data/personalize_coldstart` - Cold start personalization

### Recommendation Endpoints
- `GET /data/recommendation/{user_pseudo_id}` - Get user recommendations
- `GET /data/segments` - Get user segments
- `POST /data/coldstart` - Cold start recommendations

## 📂 Datasets

This project uses several datasets for personalization, recommendations, and analytics:

- **backend/dataset1_final.csv** (1.2GB) — Main user activity dataset (ignored by git)
- **backend/dataset2_final.csv** (2.4MB) — Supplementary user data (ignored by git)
- **backend/sample_data/** — Sample CSVs for development and testing
    - `merged_activity_transactions.csv`, `user_sessions.csv`, `user_segments.csv`
- **backend/processed_data/** — Preprocessed large datasets for fast access (ignored by git)
    - `user_sessions.csv` (1.0GB), `merged_activity_transactions.csv` (1.6GB)
- **backend/cold_start_data/** — Data and models for cold start recommendations
    - `default_trends.json`, `knn_model.pkl`, `scaler.pkl`, `user_clusters.csv`
- **backend/personalization_data/** — Personalization rules and trend data
    - `personalization_rules.json`, `personalization_trends.json`
- **backend/user_segments/** — User segmentation insights
    - `segment_insights.json`

> **Note:** Large datasets and processed files are ignored by git to keep the repository lightweight.

## 📥 Download Datasets & Large Files

Some datasets and large files are not included in the repository due to their size. Please download them from the following link:

**[Download All Datasets & Large Files (Google Drive)](YOUR_DRIVE_LINK_HERE)**

After downloading, place the files in their respective directories as described below.

## 🚫 Ignored Files & Folders

The following files and directories are ignored (not tracked by git):

- **backend/.venv/** — Python virtual environment
- **backend/__pycache__/**, **.pytest_cache/** — Python cache and test cache
- **backend/dataset1_final.csv**, **backend/dataset2_final.csv** — Large raw datasets
- **backend/processed_data/** — Preprocessed large data files
- **frontend/node_modules/** — Node.js dependencies
- **frontend/dist/** — Frontend build output
- **frontend/.vscode/**, **.idea/** — Editor settings
- ***.log**, **logs/** — Log files
- **.DS_Store**, ***.suo**, etc. — OS/editor-specific files

> See `.gitignore` files in the root, backend, and frontend directories for full details.

## 🏗️ Project Structure

```
demo/
├── backend/                          # FastAPI Backend
│   ├── app/
│   │   ├── __init__.py               # Python package initialization
│   │   ├── main.py                   # FastAPI application entry point
│   │   ├── db.py                     # Database configuration
│   │   ├── routes/                   # API route definitions
│   │   │   ├── __init__.py           # Routes package initialization
│   │   │   ├── data.py              # Data and authentication routes
│   │   │   └── personalization.py   # Personalization routes
│   │   ├── models/                   # Database models
│   │   │   ├── __init__.py           # Models package initialization
│   │   │   ├── session.py           # Session data models
│   │   │   ├── session_db.py        # Database session models
│   │   │   └── user.py              # User models
│   │   └── services/                 # Business logic
│   │       ├── __init__.py           # Services package initialization
│   │       └── ml.py                # ML and recommendation services
│   ├── config/                       # Configuration files
│   │   └── __init__.py               # Config package initialization
│   ├── tests/                        # Test files
│   │   ├── __init__.py               # Tests package initialization
│   │   └── test_api.py              # API test cases
│   ├── personalization_logic.py      # Personalization engine
│   ├── cold_start_strategy.py        # Cold start algorithms
│   ├── user_segmentation.py          # User segmentation logic
│   ├── data_preprocessing.py         # Data preprocessing utilities
│   ├── demo_preprocessing.py         # Demo preprocessing script
│   ├── accuracy_evaluation.py        # ML model accuracy evaluation
│   ├── evaluate_recommendations.py   # Recommendation evaluation
│   ├── quick_accuracy_test.py        # Quick accuracy testing
│   ├── simple_cold_start_demo.py     # Cold start demonstration
│   ├── requirements.txt              # Python dependencies
│   ├── sessions.db                   # SQLite database file
│   ├── dataset1_final.csv            # [IGNORED] Large dataset (1.2GB)
│   ├── dataset2_final.csv            # [IGNORED] Large dataset (2.4MB)
│   ├── accuracy_results.json         # Accuracy evaluation results
│   ├── quick_rating_results.json     # Quick rating results
│   ├── Untitled.ipynb                # Jupyter notebook
│   ├── README.md                     # Backend documentation
│   ├── COLD_START_STRATEGY_README.md # Cold start strategy documentation
│   ├── DATA_PREPROCESSING_README.md  # Data preprocessing documentation
│   ├── sample_data/                  # Sample datasets for development
│   │   ├── merged_activity_transactions.csv
│   │   ├── user_sessions.csv
│   │   └── user_segments.csv
│   ├── processed_data/               # [IGNORED] Preprocessed large datasets
│   │   ├── user_sessions.csv (1.0GB)
│   │   └── merged_activity_transactions.csv (1.6GB)
│   ├── cold_start_data/              # Cold start data and models
│   │   ├── default_trends.json
│   │   ├── knn_model.pkl
│   │   ├── scaler.pkl
│   │   └── user_clusters.csv
│   ├── user_segments/                # User segmentation data
│   │   └── segment_insights.json
│   ├── .venv/                        # [IGNORED] Python virtual environment
│   ├── __pycache__/                  # [IGNORED] Python cache
│   ├── .ipynb_checkpoints/           # [IGNORED] Jupyter checkpoints
│   └── .pytest_cache/                # [IGNORED] Test cache
├── frontend/                         # React Frontend
│   ├── src/
│   │   ├── App.jsx                   # Main React component
│   │   ├── App.css                   # Main styles
│   │   ├── index.css                 # Global styles
│   │   ├── main.jsx                  # React entry point
│   │   ├── personalizationApi.js     # Personalization API client
│   │   └── assets/                   # Static assets
│   │       └── react.svg
│   ├── public/                       # Public assets
│   │   └── vite.svg
│   ├── js/                           # JavaScript utilities
│   │   ├── cart.js                   # Shopping cart functionality
│   │   └── sneakers.js               # Sneakers-specific functionality
│   ├── images/                       # Product images
│   │   ├── bag/                      # Bag product images
│   │   └── [various product images]  # Product image files
│   ├── Accessories/                  # Accessories product images
│   ├── Bags/                         # Bags product images
│   ├── Clothing/                     # Clothing product images
│   ├── Jewelry/                      # Jewelry product images
│   ├── Mens/                         # Men's product images
│   ├── Shoes/                        # Shoes product images
│   ├── Watches/                      # Watches product images
│   ├── Woman/                        # Women's product images
│   ├── index.html                    # Main landing page
│   ├── login.html                    # Login page
│   ├── shop.html                     # Shop page
│   ├── categories.html               # Categories page
│   ├── men.html                      # Men's page
│   ├── women.html                    # Women's page
│   ├── clothing.html                 # Clothing page
│   ├── shoes.html                    # Shoes page
│   ├── bags.html                     # Bags page
│   ├── jewelry.html                  # Jewelry page
│   ├── accessories.html              # Accessories page
│   ├── watches.html                  # Watches page
│   ├── sale.html                     # Sale page
│   ├── account-settings.html         # Account settings page
│   ├── personalized_recommendations.js # Personalization logic
│   ├── package.json                  # Node.js dependencies
│   ├── vite.config.js                # Vite configuration
│   ├── eslint.config.js              # ESLint configuration
│   ├── .gitignore                    # Git ignore rules
│   ├── README.md                     # Frontend documentation
│   ├── node_modules/                 # [IGNORED] Node.js dependencies
│   ├── dist/                         # [IGNORED] Build output
│   └── .vscode/                      # [IGNORED] Editor settings
├── personalization_data/             # Personalization data files
│   ├── personalization_rules.json    # Personalization rules
│   └── personalization_trends.json   # Trend data
├── .venv/                            # [IGNORED] Root virtual environment
├── .vscode/                          # [IGNORED] Editor settings
├── accuracy_results.json             # Global accuracy results
├── quick_rating_results.json         # Global rating results
├── Drive Doc - Hyper-Personalized Landing Page Generator Agent.pdf # Project documentation
└── README.md                         # This file
```

**Legend:**
- `├──` = Directory with more items below
- `└──` = Last item in a directory
- `│   ` = Vertical line showing hierarchy
- `[IGNORED]` = Files/folders not tracked by git

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database Configuration
DATABASE_URL=sqlite:///./sessions.db

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ML Model Configuration
MODEL_PATH=./models/
DATA_PATH=./data/

# API Configuration
CORS_ORIGINS=["http://localhost:5173", "http://127.0.0.1:5173"]
```

### Backend Configuration

The backend uses SQLite by default. For production, consider using PostgreSQL:

```python
# In backend/app/db.py
DATABASE_URL = "postgresql://user:password@localhost/dbname"
```

## 🧪 Testing

### Backend Testing
```bash
cd backend
python -m pytest tests/
```

### Frontend Testing
```bash
cd frontend
npm test
```

### API Testing
```bash
# Test with curl
curl -X GET "http://localhost:8000/data/sessions"

# Test with the interactive docs
# Visit http://localhost:8000/docs
```

## 📈 Performance Optimization

### Backend Optimization
- Use connection pooling for database connections
- Implement caching with Redis
- Optimize ML model loading
- Use async/await for I/O operations

### Frontend Optimization
- Implement code splitting
- Use React.memo for expensive components
- Optimize bundle size with tree shaking
- Implement lazy loading for images

## 🔒 Security Considerations

- All passwords are hashed using bcrypt
- CORS is configured for development
- Input validation on all API endpoints
- SQL injection protection via SQLAlchemy
- Rate limiting should be implemented for production

## 🚀 Deployment

### Backend Deployment
```bash
# Production build
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Deployment
```bash
# Build for production
npm run build

# Serve static files
npm install -g serve
serve -s dist -l 3000
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Review the error logs in the terminal
3. Check the API documentation at http://localhost:8000/docs
4. Open an issue on GitHub with detailed error information

## 📊 Monitoring & Logs

### Backend Logs
- Check terminal output for FastAPI logs
- Database logs are in `sessions.db`
- ML model logs are in the console

### Frontend Logs
- Check browser developer tools (F12)
- Vite dev server logs in terminal
- Network tab for API calls

---

**Happy Coding! 🎉**

For more detailed information about specific components, check the individual README files in the `backend/` and `frontend/` directories.
