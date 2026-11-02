
import hashlib, os
SALT = os.getenv("PSEUDO_SALT", "please-change-me")
def pseudonymize_id(patient_id: str) -> str:
    if not patient_id:
        return None
    h = hashlib.sha256()
    h.update(SALT.encode("utf-8"))
    h.update(patient_id.encode("utf-8"))
    return h.hexdigest()[:24]
