# CredibilityScan Backend

This directory contains the FastAPI backend application for CredibilityScan. It serves as the API for processing text and video content using machine learning models.

## Project Structure

```
backend/
├── app.py                      # Main FastAPI application
├── analysis_helpers.py         # Helper functions for text analysis
├── cookies.txt                 # (Potentially for web scraping, review if sensitive)
├── deepfake_detector.py        # Deepfake detection logic
├── GCustomSearch.py            # Google Custom Search API integration
├── GEMINI.md                   # Project context (can be moved/summarized if desired)
├── model_loader.py             # Logic for loading ML models
├── requirements.txt            # Python dependencies
├── stopwords.py                # Stopwords for NLP
├── test_tokenizer.py           # Testing related to tokenization
├── Models/                     # Machine Learning models (YOLOv8, MobileNetV2, XGBoost etc.)
├── routers/                    # FastAPI routers (e.g., app_video.py)
├── __pycache__/                 # Python cache files (ignored by .gitignore)
└── README.md                   # This file
```

## Setup and Local Development

1.  **Clone the repository:**
    ```bash
    git clone <your-backend-repo-url>
    cd credibilityscan-backend
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    *   **Windows:** `.\venv\Scripts\activate`
    *   **macOS/Linux:** `source venv/bin/activate`
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Environment Variables:**
    If your application uses environment variables (e.g., for API keys), create a `.env` file in the `backend/` directory based on a `.env.example` (if provided), or set them directly in your shell session.
    *   `SECRET_KEY` for session middleware
    *   `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` for Google Custom Search API

6.  **Run the application:**
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will be available at `http://localhost:8000`.

## Deployment

This backend is designed for deployment on Python-friendly cloud hosts like Render or Railway.

### Render

1.  **Sign up/Log in:** Create an account or log in to [Render](https://render.com/).
2.  **New Web Service:** In your Render dashboard, click "New" -> "Web Service".
3.  **Connect to Git:** Link your GitHub repository (this `backend` repository).
4.  **Configure:**
    *   **Name:** `credibilityscan-api` (or your preferred name).
    *   **Runtime:** `Python 3`.
    *   **Build Command:** `pip install -r requirements.txt`.
    *   **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`.
5.  **Environment Variables:** Add any required environment variables (e.g., `SECRET_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`).
6.  **Deploy:** Render will automatically build and deploy your application.

### Railway

1.  **Sign up/Log in:** Create an account or log in to [Railway](https://railway.app/).
2.  **New Project:** In your Railway dashboard, click "New Project" -> "Deploy from GitHub Repo".
3.  **Connect to Git:** Link your GitHub repository (this `backend` repository).
4.  **Configure:** Railway typically auto-detects most settings.
    *   **Build Command:** Confirm it detects `pip install -r requirements.txt`.
    *   **Start Command:** Confirm it detects `uvicorn app:app --host 0.0.0.0 --port $PORT`.
5.  **Environment Variables:** Add any required environment variables.
6.  **Deploy:** Railway will build and deploy your service.
