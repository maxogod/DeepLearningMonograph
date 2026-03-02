import os
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class Config:
    def __init__(self):
        env = os.environ.get("ENVIRONMENT")
        self.environment = Environment.DEVELOPMENT
        if env == "production":
            self.environment = Environment.PRODUCTION
