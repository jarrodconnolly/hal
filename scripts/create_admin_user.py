import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from argon2 import PasswordHasher
from src.hal.config import MONGO_URI, MONGO_DB_NAME, USERS_COLLECTION

async def create_admin_user():
    # Connect to MongoDB
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    
    # Drop users collection
    await db[USERS_COLLECTION].drop()
    print(f"Dropped {USERS_COLLECTION} collection")
    
    # Hash admin password
    ph = PasswordHasher()
    admin_password = "12345"
    hashed_password = ph.hash(admin_password)
    
    # Admin user document
    admin_user = {
        "username": "jarrod",
        "password": hashed_password,
        "email": "jarrod@nestedquotes.ca"
    }
    
    # Insert admin user
    await db[USERS_COLLECTION].insert_one(admin_user)
    print(f"Created admin user: {admin_user['username']}")

if __name__ == "__main__":
    asyncio.run(create_admin_user())