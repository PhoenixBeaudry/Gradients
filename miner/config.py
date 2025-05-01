from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


T = TypeVar("T", bound=BaseModel)
