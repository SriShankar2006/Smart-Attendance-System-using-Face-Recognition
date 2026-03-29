# ============================================================
#  MODULE 2 — FACE ENCODING  (PREPROCESSING)
#  Smart Attendance System using Face Recognition
# ============================================================
"""
Encodes only newly registered student face images into 128-d face descriptors
and appends them to the existing encodings file.

Workflow:
    1. Loads existing encodings from encodings.pkl (if present).
    2. Reads images from dataset/<ID>_<Name>/ folders.
    3. Skips students whose IDs are already encoded.
    4. Detects faces using the HOG model (faster, CPU-only).
    5. Computes a 128-dimensional encoding per detected face.
    6. Appends new encodings to encodings.pkl.

Run:
    python module2_encode_faces.py

Prerequisites:
    python module1_register_face.py   (students.json + dataset/ must exist)

Next step:
    python module3_train_model.py
"""

import face_recognition
import os
import pickle
import json
import sys
import time

import numpy as np

# ── File / Directory Constants ────────────────────────────────
DATASET_DIR    = "dataset"
ENCODINGS_FILE = "encodings.pkl"
STUDENTS_FILE  = "students.json"

SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")   # image formats to process

# ── ANSI Colors ───────────────────────────────────────────────
R      = "\033[0m"   # Reset
B      = "\033[1m"   # Bold
DM     = "\033[2m"   # Dim
CY     = "\033[96m"  # Cyan
GR     = "\033[92m"  # Green
YL     = "\033[93m"  # Yellow
RD     = "\033[91m"  # Red
WH     = "\033[97m"  # White
BG_BL  = "\033[44m"  # Blue background
BG_GR  = "\033[42m"  # Green background


# ══════════════════════════════════════════════════════════════
#  Terminal UI helpers
# ══════════════════════════════════════════════════════════════

def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def banner() -> None:
    """Print the module header banner."""
    clear()
    print(f"\n{B}{'═'*58}{R}")
    print(f"{B}   Smart Attendance System{R}")
    print(f"{B}   MODULE 2 — Face Encoding & Preprocessing{R}")
    print(f"{B}{'═'*58}{R}\n")


def ok(msg: str)   -> None: print(f"  {GR}✔{R}  {msg}")
def warn(msg: str) -> None: print(f"  {YL}⚠{R}  {msg}")
def err(msg: str)  -> None: print(f"  {RD}✖{R}  {msg}")
def info(msg: str) -> None: print(f"  {CY}ℹ{R}  {msg}")


def spinner_msg(msg: str, duration: float = 0.6) -> None:
    """Display a spinner animation for *duration* seconds, then print a tick."""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end    = time.time() + duration
    i      = 0
    while time.time() < end:
        print(f"\r  {CY}{frames[i % len(frames)]}{R}  {msg}", end="", flush=True)
        time.sleep(0.07)
        i += 1
    print(f"\r  {GR}✔{R}  {msg}{' '*10}")


def progress_bar(done: int, total: int, name: str = "", width: int = 28) -> None:
    """Render an inline progress bar to stdout."""
    filled = int(width * done / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(100 * done / total)
    color  = GR if pct == 100 else YL if pct >= 50 else CY
    label  = f"{name[:16]:<16}" if name else ""
    print(f"\r  {label}  {color}[{bar}]{R} {pct:3d}%  {done}/{total}",
          end="", flush=True)


def section_header(title: str) -> None:
    print(f"\n  {BG_BL}{WH}  {title}  {R}\n")


# ══════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════

def load_students() -> dict:
    """
    Load students.json.  Exits with an error if the file does not exist
    (the user must run Module 1 first).
    """
    if not os.path.exists(STUDENTS_FILE):
        err("students.json not found — run Module 1 first.")
        sys.exit(1)
    with open(STUDENTS_FILE) as f:
        return json.load(f)


def load_existing_encodings() -> tuple:
    """
    Load encodings.pkl if it exists.
    Returns (encodings, names, ids) — all empty lists if file not found.
    """
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data["encodings"], data["names"], data["ids"]
    return [], [], []


# ══════════════════════════════════════════════════════════════
#  Core encoding logic
# ══════════════════════════════════════════════════════════════

def encode_all() -> None:
    """
    Iterate over all student dataset folders, compute face encodings only
    for students not already present in encodings.pkl, and append results.

    Each entry in the output file contains:
        - encodings  : list of 128-d numpy arrays
        - names      : list of student names (parallel to encodings)
        - ids        : list of student IDs  (parallel to encodings)
    """
    banner()
    students = load_students()
    info(f"Students registered: {', '.join(f'{B}{v}{R}' for v in students.values())}\n")

    # ── Load existing encodings ──────────────────────────────
    known_encodings, known_names, known_ids = load_existing_encodings()
    existing_ids = set(known_ids)

    if existing_ids:
        info(f"Already encoded: {len(existing_ids)} student(s) — will skip these.\n")
    else:
        info("No existing encodings found — encoding all students.\n")

    section_header("Generating Face Encodings")

    summary:      dict = {}
    new_students: int  = 0

    folders = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    if not folders:
        err("No dataset folders found. Run Module 1 first.")
        sys.exit(1)

    for folder_name in folders:
        parts  = folder_name.split("_", 1)
        sid    = parts[0]
        sname  = parts[1] if len(parts) > 1 else parts[0]
        folder = os.path.join(DATASET_DIR, folder_name)

        # ── Skip already-encoded students ────────────────────
        if sid in existing_ids:
            warn(f"{sname} (ID: {sid}) already encoded → skipping")
            continue

        images   = [f for f in os.listdir(folder)
                    if f.lower().endswith(SUPPORTED_EXTS)]
        ok_count = 0
        new_students += 1

        print(f"\n  {CY}{B}{sname}{R}  {DM}(ID: {sid}){R}  —  {len(images)} image(s)")

        for i, img_file in enumerate(images):
            progress_bar(i + 1, len(images), sname)
            try:
                img  = face_recognition.load_image_file(
                    os.path.join(folder, img_file)
                )
                locs = face_recognition.face_locations(img, model="hog")
                if not locs:
                    continue
                enc  = face_recognition.face_encodings(img, locs)[0]
                known_encodings.append(enc)
                known_names.append(sname)
                known_ids.append(sid)
                ok_count += 1
            except Exception:
                # Skip corrupted / unreadable images silently
                pass

        print()  # newline after inline progress bar
        summary[sname] = {"total": len(images), "ok": ok_count}
        status = (
            f"{GR}✔ {ok_count} encodings{R}"
            if ok_count > 0
            else f"{RD}✖ 0 encodings — check image quality{R}"
        )
        print(f"         → {status}")

    # ── Nothing new to encode ────────────────────────────────
    if new_students == 0:
        print()
        warn("No new students to encode. All registered students are already in encodings.pkl.")
        print(f"\n  {DM}Add a new student via Module 1, then re-run this module.{R}\n")
        sys.exit(0)

    # ── Persist encodings ────────────────────────────────────
    section_header("Saving Encodings")
    spinner_msg("Writing encodings.pkl ...", duration=0.8)

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(
            {"encodings": known_encodings,
             "names":     known_names,
             "ids":       known_ids},
            f,
        )

    # ── Summary table ────────────────────────────────────────
    print(f"\n  {DM}{'─'*48}{R}")
    print(f"  {'Name':<20} {'Images':>8} {'Encoded':>9} {'Quality':>10}")
    print(f"  {DM}{'─'*48}{R}")

    for sname, s in summary.items():
        rate   = (s["ok"] / s["total"] * 100) if s["total"] else 0
        status = (
            f"{GR}Good{R}"   if rate >= 80
            else f"{YL}Low{R}"    if rate > 0
            else f"{RD}Failed{R}"
        )
        print(f"  {WH}{sname:<20}{R} {s['total']:>8} {s['ok']:>9}    {status}")

    print(f"  {DM}{'─'*48}{R}")
    total_enc = len(known_encodings)
    print(f"\n  {GR}{B}✔  {len(summary)} new student(s) encoded.  "
          f"{total_enc} total encodings in encodings.pkl{R}")
    print(f"\n  {DM}Next step →  python module3_train_model.py{R}\n")


if __name__ == "__main__":
    encode_all()