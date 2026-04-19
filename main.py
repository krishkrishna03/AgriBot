from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from pymongo import MongoClient
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
import os
from bson import ObjectId

from nlp_model import get_chatbot_response, reload_model

# configuration
SECRET_KEY = "super_secret_key_for_agribot"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# MongoDB connect (default localhost:27017 for compass)
MONGO_DETAILS = "mongodb+srv://patnalasrikrishnasai:Login@cluster0.igezekd.mongodb.net/agribot"
try:
    client = MongoClient(MONGO_DETAILS, serverSelectionTimeoutMS=5000)
    db = client.agribot_db
    users_collection = db.get_collection("users")
    chats_collection = db.get_collection("chats")
except Exception as e:
    print("Warning: Could not connect to MongoDB. Make sure MongoDB is running.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title="AgriBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class UserCreate(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    message: str

class ChatHistoryResponse(BaseModel):
    user_msg: str
    bot_msg: str
    timestamp: str

# Helpers
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = users_collection.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user

# Routes
@app.post("/register")
async def register(user: UserCreate):
    existing_user = users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    users_collection.insert_one({"username": user.username, "hashed_password": hashed_password})
    return {"message": "User created successfully"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat")
async def chat(message_data: ChatMessage, current_user: dict = Depends(get_current_user)):
    user_msg = message_data.message
    bot_msg = get_chatbot_response(user_msg)
    
    # Save to history
    chat_doc = {
        "username": current_user["username"],
        "user_msg": user_msg,
        "bot_msg": bot_msg,
        "timestamp": datetime.utcnow()
    }
    chats_collection.insert_one(chat_doc)
    
    return {"reply": bot_msg}

@app.get("/history")
async def get_history(current_user: dict = Depends(get_current_user)):
    cursor = chats_collection.find({"username": current_user["username"]}).sort("timestamp", 1)
    history = []
    for doc in cursor:
        history.append({
            "user_msg": doc["user_msg"],
            "bot_msg": doc["bot_msg"],
            "timestamp": doc["timestamp"].isoformat()
        })
    return {"history": history}

@app.get("/model/info")
async def model_info():
    """Returns metadata about the currently loaded model."""
    import json, os
    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "meta.json")
    if not os.path.exists(meta_path):
        return {"status": "no model found", "message": "Run python train_model.py first"}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["status"] = "loaded"
    return meta

@app.post("/admin/reload-model")
async def admin_reload_model(current_user: dict = Depends(get_current_user)):
    """
    Hot-reload the model without restarting the server.
    Useful after running python train_model.py with new data.
    """
    success = reload_model()
    if success:
        return {"status": "success", "message": "Model reloaded successfully from disk."}
    raise HTTPException(
        status_code=503,
        detail="Model reload failed. Ensure train_model.py has been run successfully."
    )

# Serve static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.api_route("/", methods=["GET", "HEAD"])
async def serve_frontend(request: Request):
    # Let health checks use HEAD without triggering a 405 on the root path.
    if request.method == "HEAD":
        return Response(status_code=200)
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    import os

    if not os.path.exists("static"):
        os.makedirs("static")

    port = int(os.environ.get("PORT", 8000))  # IMPORTANT
    uvicorn.run("main:app", host="0.0.0.0", port=port)
