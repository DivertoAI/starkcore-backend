from app.models.user import User

def create_user(user: User):
    # For now, just echo the user back
    return {
        "message": "User created successfully",
        "user": user.dict()
    }