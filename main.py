from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime, timezone
import os
import uuid

from classifier import train as train_model, predict as classify_note, get_categories, trained_status
from reminder import extract_reminder_info

DATABASE_URL = "sqlite:///./notes_new.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Note(Base):
    __tablename__ = "notes"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    content = Column(Text, nullable=False)
    audio_path = Column(String, nullable=True)
    category = Column(String, nullable=True, default="other")
    is_reminder = Column(Boolean, default=False)
    reminder_date = Column(String, nullable=True)
    reminder_time = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Notes API")
app.mount("/static", StaticFiles(directory="static"), name="static")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("static/index.html")

@app.post("/api/notes")
async def create_note(
    content: str = Form(...),
    audio: UploadFile = File(None),
    db = None
):
    if db is None:
        db = SessionLocal()
    try:
        audio_path = None
        if audio and audio.filename:
            os.makedirs("uploads", exist_ok=True)
            ext = os.path.splitext(audio.filename)[1]
            audio_path = f"uploads/{uuid.uuid4()}{ext}"
            with open(audio_path, "wb") as f:
                f.write(await audio.read())
            audio_path = os.path.basename(audio_path)

        # Classify
        cat_result = classify_note(content)
        category = cat_result.get("category", "other") if cat_result.get("trained") else "other"

        # Reminder scan
        rem = extract_reminder_info(content)
        is_reminder = rem["is_reminder"]
        reminder_date = rem["reminder_date"]
        reminder_time = rem["time"]

        note = Note(
            content=content,
            audio_path=audio_path,
            category=category,
            is_reminder=is_reminder,
            reminder_date=reminder_date,
            reminder_time=reminder_time,
        )
        db.add(note)
        db.commit()
        db.refresh(note)
        return {
            "id": note.id,
            "content": note.content,
            "audio_path": note.audio_path,
            "category": note.category,
            "is_reminder": note.is_reminder,
            "reminder_date": note.reminder_date,
            "reminder_time": note.reminder_time,
            "created_at": note.created_at.isoformat(),
        }
    finally:
        db.close()

@app.get("/api/notes")
async def list_notes():
    db = SessionLocal()
    try:
        notes = db.query(Note).order_by(Note.created_at.desc()).all()
        return [
            {
                "id": n.id,
                "content": n.content,
                "audio_path": n.audio_path,
                "category": n.category,
                "is_reminder": n.is_reminder,
                "reminder_date": n.reminder_date,
                "reminder_time": n.reminder_time,
                "created_at": n.created_at.isoformat(),
                "updated_at": n.updated_at.isoformat(),
            }
            for n in notes
        ]
    finally:
        db.close()

@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    db = SessionLocal()
    try:
        note = db.query(Note).filter(Note.id == note_id).first()
        if not note:
            raise HTTPException(404, "Note not found")
        if note.audio_path and os.path.exists(note.audio_path):
            os.remove(note.audio_path)
        db.delete(note)
        db.commit()
        return {"ok": True}
    finally:
        db.close()

@app.get("/api/backup")
async def backup_notes():
    db = SessionLocal()
    try:
        notes = db.query(Note).order_by(Note.created_at.asc()).all()
        return {
            "backup_date": datetime.now(timezone.utc).isoformat(),
            "notes": [
                {
                    "id": n.id,
                    "content": n.content,
                    "audio_path": n.audio_path,
                    "category": n.category,
                    "is_reminder": n.is_reminder,
                    "reminder_date": n.reminder_date,
                    "reminder_time": n.reminder_time,
                    "created_at": n.created_at.isoformat(),
                    "updated_at": n.updated_at.isoformat(),
                }
                for n in notes
            ]
        }
    finally:
        db.close()

@app.get("/api/health")
async def health():
    return {"status": "ok"}

# ── Classification endpoints ────────────────────────────────────────────────

@app.post("/api/train")
async def train_classifier(
    texts: str = Form(...),
    labels: str = Form(...),
):
    """Train the classifier.
    texts:  JSON array of note strings  e.g. '["buy milk","submit report"]'
    labels: JSON array of category names e.g. '["shopping","work"]'
    """
    import json
    try:
        text_list = json.loads(texts)
        label_list = json.loads(labels)
    except json.JSONDecodeError:
        raise HTTPException(400, "texts and labels must be valid JSON arrays")

    if len(text_list) != len(label_list):
        raise HTTPException(400, "texts and labels must have the same length")
    if len(text_list) < 2:
        raise HTTPException(400, "need at least 2 examples to train")

    train_model(text_list, label_list)
    return {
        "ok": True,
        "categories": get_categories(),
        "note_count": len(text_list),
    }

@app.post("/api/classify")
async def classify_text(
    content: str = Form(...),
):
    """Classify a single text without saving it."""
    result = classify_note(content)
    return result

@app.get("/api/categories")
async def list_categories():
    return {"categories": get_categories(), "trained": trained_status()}

# ── Reminder endpoints ───────────────────────────────────────────────────────

@app.get("/api/reminders")
async def list_reminders():
    """Return notes flagged as reminders, sorted by reminder_date."""
    db = SessionLocal()
    try:
        notes = db.query(Note).filter(Note.is_reminder == True).order_by(Note.reminder_date.asc()).all()
        return [
            {
                "id": n.id,
                "content": n.content,
                "category": n.category,
                "reminder_date": n.reminder_date,
                "reminder_time": n.reminder_time,
                "created_at": n.created_at.isoformat(),
            }
            for n in notes
        ]
    finally:
        db.close()
