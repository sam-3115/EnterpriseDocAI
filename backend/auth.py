import bcrypt
from typing import Optional, Dict, Any
from backend.database import get_user_by_username, create_user

def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt with salt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate username and password against SQLite database."""
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
        "created_at": user["created_at"]
    }

def register_user(username: str, password: str, role: str = "user") -> Dict[str, Any]:
    """Register a new user in SQLite database."""
    clean_username = username.strip()
    if not clean_username:
        raise ValueError("Username cannot be empty")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters long")
    
    clean_role = role.strip().lower()
    if clean_role not in ["admin", "user"]:
        clean_role = "user"
        
    existing = get_user_by_username(clean_username)
    if existing:
        raise ValueError(f"Username '{clean_username}' already exists")
        
    pwd_hash = hash_password(password)
    user_id = create_user(clean_username, pwd_hash, clean_role)
    return {
        "id": user_id,
        "username": clean_username,
        "role": clean_role
    }

