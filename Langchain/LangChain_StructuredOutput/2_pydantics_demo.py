from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class student(BaseModel):
    name : str = "Ayush"
    age : Optional[int] = None
    email : EmailStr
    cgpa : float = Field(gt=0, lt=10)

new_student = {"age" : "22", "email" : "ayushuniyal@gmail.com", "cgpa" : 1}

student = student(**new_student)
print(student)