from pydantic import BaseModel
from typing import List


class StudentProfile(BaseModel):
    name: str = "John Doe"
    grade: int = 10
    subject: str = "Financial Literacy"
    interests: List[str] = ["Video Games", "Computers", "F1 Racing"]
