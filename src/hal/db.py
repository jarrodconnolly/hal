from motor.motor_asyncio import AsyncIOMotorClient
from .config import MONGO_URI, MONGO_DB_NAME, USERS_COLLECTION
from argon2 import PasswordHasher, exceptions

_client = None
_ph = PasswordHasher()

async def get_db():
    """Get the MongoDB database handle for HAL."""
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URI)
    return _client[MONGO_DB_NAME]

async def authenticate(username: str, password: str) -> dict | None:
    """Authenticate a user by username and password, return user dict or None."""
    db = await get_db()
    user = await db[USERS_COLLECTION].find_one({"username": username})
    
    # Dummy hash for timing safety - used if no user found
    dummy_hash = "$argon2id$v=19$m=65536,t=3,p=4$nxP1H6wz2Ab4kLWH3RA/Cg$7GjWa9MU1SYjUWAxWVWU/GHglXBQWLhHS/hc4Xdth70"
    stored_hash = user["password"] if user else dummy_hash
    
    try:
        # Always verify - constant time whether user exists or not
        _ph.verify(stored_hash, password.encode("utf-8"))
        return user if user else None  # Only return user if found AND hash matches
    except exceptions.VerifyMismatchError:
        return None  # Wrong password
    except exceptions.InvalidHashError:
        return None  # Bad hash format (shouldn’t happen)