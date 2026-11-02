
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PatientBase(BaseModel):
    patient_id: Optional[str]
    age: Optional[int]
    gender: Optional[str]
    diagnosis: Optional[str]
    admission_date: Optional[datetime]
    discharge_date: Optional[datetime]
    outcome: Optional[str]
    death_date: Optional[datetime]
    followup_days: Optional[int]
    cause: Optional[str]

class PatientCreate(PatientBase):
    pass

class Patient(PatientBase):
    id: int
    class Config:
        orm_mode = True
