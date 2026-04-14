# AgriBot - AI Agriculture Assistant

A premium chatbot built for agriculture with a complete ChatGPT/Claude-like interface, user authentication, MongoDB integration, and an NLP/KNN-powered backend.

## Tech Stack
- **Frontend**: HTML5, Vanilla JavaScript, CSS3 (Premium Glassmorphism & Dark Mode)
- **Backend API**: Python 3, FastAPI, Uvicorn
- **AI Model**: Scikit-Learn (KNN Classifier), NLTK, TF-IDF Vectorizer
- **Database**: MongoDB Compass (PyMongo)
- **Authentication**: JWT Bearer Tokens, Bcrypt Hashing

## Features
- **ChatGPT-like Interface**: Sleek dark-mode aesthetic with animations and responsive layout.
- **NLP & KNN Model**: Chat logic resolves agriculture questions using K-Nearest Neighbors vector-based text classification.
- **Account Management**: Register and login system.
- **Chat History**: Sidebar displays previous chats dynamically loaded from the database.
- **MongoDB Database**: Connected locally directly to a MongoDB Compass instance (`mongodb://localhost:27017`).

## How to Run

### 1. Requirements
Ensure you have the following installed on your Windows system:
- Python 3.9+
- Pip
- MongoDB Compass / MongoDB Server (running on default port 27017)

### 2. Installation
Open your terminal (PowerShell or CMD) in the `e:\agribot` directory and install the required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Start the Server
Run the FastAPI application via uvicorn:
```bash
python -m uvicorn main:app --reload
```
or 
```bash
uvicorn main:app --reload
```

### 4. Access the Application
The backend serves both the API and the Frontend. Open your browser and go to:
[http://localhost:8000](http://localhost:8000)

## Design Notes
The UI employs a premium dark theme (`#343541`, `#202123`), smooth flexbox-based layout for the sidebar and main chat surface, micro-animations for message typing and hover states, and Lucide icons for high-quality SVG symbology.
