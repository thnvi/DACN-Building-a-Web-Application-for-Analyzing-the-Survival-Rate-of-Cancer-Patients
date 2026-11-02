
from sqlalchemy.orm import Session
from . import models, schemas
from .utils import pseudonymize_id

def get_patients(db: Session, skip: int = 0, limit: int = 1000):
    return db.query(models.Patient).offset(skip).limit(limit).all()

def create_patient(db: Session, patient: schemas.PatientCreate):
    data = patient.dict()
    if data.get("patient_id"):
        data["patient_id"] = pseudonymize_id(data["patient_id"])
    db_obj = models.Patient(**data)
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj
