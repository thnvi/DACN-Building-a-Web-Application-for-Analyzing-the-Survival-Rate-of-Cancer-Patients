
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import crud, schemas
from ..database import SessionLocal

router = APIRouter(prefix="/patients", tags=["patients"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/", response_model=schemas.Patient)
def create_patient(p: schemas.PatientCreate, db: Session = Depends(get_db)):
    return crud.create_patient(db, p)

@router.get("/", response_model=list[schemas.Patient])
def read_patients(skip: int = 0, limit: int = 1000, db: Session = Depends(get_db)):
    return crud.get_patients(db, skip, limit)
