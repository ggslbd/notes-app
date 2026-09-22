# Notes App

Voice or write notes from your Android phone — with ML classification and keyword-triggered reminders. Backed by a FastAPI + SQLite backend.

## What it does

- **Write or record notes** — type text or use the mic button to record voice (uploaded as audio file)
- **Auto-classification** — each note gets a category (`personal`, `work`, `health`, `shopping`, `idea`, `other`) via a scikit-learn TF-IDF + LogisticRegression model. Train it on your own labeled notes.
- **Reminder detection** — notes containing `remind`, `todo`, `don't forget`, etc. are flagged. Dates and times mentioned in the text (e.g. "tomorrow at 3pm", "2026-09-25") are extracted and stored.
- **Backup** — download all notes as JSON at any time.

## Quick start

**Windows:**

```bash
cd notes-app
.\run.bat
```

`run.bat` creates the venv, installs deps, and starts the server on `http://localhost:8082`.

**Manual:**

```bash
cd notes-app
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8082
```

Open `http://localhost:8082` in a browser. On Android, use your PC's LAN IP (e.g. `http://192.168.1.5:8082`) from Chrome.

## Usage

### Write / record a note

1. Type in the text area or tap 🎤 to record voice
2. Notes mentioning `remind`, `todo`, `don't forget`, etc. are auto-flagged as reminders
3. Tap **Save Note**

### Train the classifier

1. Tap **✏️ Label & Train**
2. Assign a category to each note from the dropdown
3. Optionally add example rows directly
4. Tap **Train Model** — the model retrains instantly
5. New notes are auto-classified from then on

Default categories: `personal`, `work`, `health`, `shopping`, `idea`, `other`. Add or remove by labeling.

### Backup

Tap **Download Backup** to get a `notes-backup-YYYY-MM-DD.json` file. Use the `/api/backup` endpoint to restore later.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Mobile-friendly web UI |
| `POST` | `/api/notes` | Create note (form: `content`, optional `audio` file) |
| `GET` | `/api/notes` | List all notes, newest first |
| `DELETE` | `/api/notes/{id}` | Delete a note |
| `GET` | `/api/backup` | Full JSON dump for backup/restore |
| `POST` | `/api/train` | Train classifier — form: `texts` (JSON array), `labels` (JSON array) |
| `POST` | `/api/classify` | Classify a single text — form: `content` |
| `GET` | `/api/categories` | List categories + trained status |
| `GET` | `/api/reminders` | Notes flagged as reminders, sorted by date |
| `GET` | `/api/health` | Liveness check |

## Project structure

```
notes-app/
├── main.py              # FastAPI backend
├── classifier.py        # ML model (TF-IDF + LogisticRegression)
├── reminder.py          # Keyword + date/time extraction
├── requirements.txt     # Python deps
├── run.bat              # Windows starter
├── static/
│   └── index.html       # Mobile UI
├── .gitignore
└── .venv/               # Python venv (gitignored)
```

## Tech stack

- **Backend:** Python, FastAPI, SQLite (SQLAlchemy)
- **ML:** scikit-learn (TF-IDF + LogisticRegression)
- **Frontend:** Vanilla HTML/CSS/JS, mobile-friendly
- **Voice:** MediaRecorder API (browser) → audio file upload
