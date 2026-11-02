
from sqlalchemy import Column, Integer, String, DateTime
from .database import Base

class Patient(Base):
    __tablename__ = 'patients'
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    diagnosis = Column(String, nullable=True)
    admission_date = Column(DateTime, nullable=True)
    discharge_date = Column(DateTime, nullable=True)
    outcome = Column(String, nullable=True)
    death_date = Column(DateTime, nullable=True)
    followup_days = Column(Integer, nullable=True)
    cause = Column(String, nullable=True)
