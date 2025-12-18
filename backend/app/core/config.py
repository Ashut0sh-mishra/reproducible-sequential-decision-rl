from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Battery RL Backend"

    REDIS_BROKER: str = "redis://localhost:6379/0"
    REDIS_BACKEND: str = "redis://localhost:6379/1"

    class Config:
        env_file = ".env"

settings = Settings()
