# ============================================================
#  MODULE 1 — FACE REGISTRATION
#  Smart Attendance System using Face Recognition
#  Author: Team Project | Python Lab
# ============================================================
"""
Registers new students by capturing face images via webcam.

Workflow:
    1. Enter Student ID and Name.
    2. Webcam opens and captures face images automatically.
    3. Images are saved to dataset/<ID>_<Name>/.
    4. Student record is persisted in students.json.

Run:
    python module1_register_face.py

Next step:
    python module2_encode_faces.py
"""

import cv2
import os
import json
import shutil
import time

# ── File / Directory Constants ────────────────────────────────
DATASET_DIR   = "dataset"
STUDENTS_FILE = "students.json"
MIN_CAPTURE   = 10          # minimum images required per student
TARGET_CAPTURE = 30         # target number of images to capture
FACE_SIZE     = (160, 160)  # pixel dimensions saved per face crop

os.makedirs(DATASET_DIR, exist_ok=True)

# ── ANSI Color Palette ────────────────────────────────────────
R      = "\033[0m"   # Reset
B      = "\033[1m"   # Bold
DM     = "\033[2m"   # Dim
CY     = "\033[96m"  # Cyan
GR     = "\033[92m"  # Green
YL     = "\033[93m"  # Yellow
RD     = "\033[91m"  # Red
BL     = "\033[94m"  # Blue
WH     = "\033[97m"  # White
BG_BL  = "\033[44m"  # Blue background
BG_GR  = "\033[42m"  # Green background


# ══════════════════════════════════════════════════════════════
#  Terminal UI helpers
# ══════════════════════════════════════════════════════════════

def clear() -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def banner() -> None:
    """Print the module header banner."""
    clear()
    print(f"\n{CY}{B}{'═'*58}{R}")
    print(f"{CY}{B}   ██████╗ ███████╗ ██████╗{R}")
    print(f"{CY}{B}   ██╔══██╗██╔════╝██╔════╝{R}")
    print(f"{CY}{B}   ██████╔╝█████╗  ██║     {R}")
    print(f"{CY}{B}   ██╔══██╗██╔══╝  ██║     {R}")
    print(f"{CY}{B}   ██║  ██║███████╗╚██████╗{R}")
    print(f"{CY}{B}   Smart Attendance System  {R}")
    print(f"{CY}{B}{'═'*58}{R}")
    print(f"{DM}   MODULE 1 — Face Registration{R}\n")


def section(title: str) -> None:
    """Print a styled section divider."""
    print(f"\n{BL}{B}  ┌─ {title} {'─'*(44-len(title))}┐{R}")


def ok(msg: str)   -> None: print(f"  {GR}✔{R}  {msg}")
def warn(msg: str) -> None: print(f"  {YL}⚠{R}  {msg}")
def err(msg: str)  -> None: print(f"  {RD}✖{R}  {msg}")
def info(msg: str) -> None: print(f"  {CY}ℹ{R}  {msg}")
def step(n: int, msg: str) -> None:
    print(f"\n  {BG_BL}{WH} STEP {n} {R} {B}{msg}{R}")


def progress_bar(current: int, total: int, width: int = 30) -> None:
    """Render an inline progress bar to stdout."""
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(100 * current / total)
    color  = GR if pct >= 80 else YL if pct >= 40 else CY
    print(f"\r  {color}[{bar}]{R} {B}{pct:3d}%{R}  {current}/{total} images",
          end="", flush=True)


# ══════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════

def load_students() -> dict:
    """Load the students registry from disk. Returns empty dict if absent."""
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE) as f:
            return json.load(f)
    return {}


def save_students(students: dict) -> None:
    """Persist the students registry to disk."""
    with open(STUDENTS_FILE, "w") as f:
        json.dump(students, f, indent=2)


def show_registered(students: dict) -> None:
    """Display a formatted table of currently registered students."""
    section("Registered Students")
    if not students:
        warn("No students registered yet.")
    else:
        print(f"  {DM}  {'ID':<10} {'Name':<20} {'Images'}{R}")
        print(f"  {DM}  {'─'*8} {'─'*18} {'─'*10}{R}")
        for sid, sname in students.items():
            folder = os.path.join(DATASET_DIR, f"{sid}_{sname}")
            count  = len(os.listdir(folder)) if os.path.exists(folder) else 0
            dot    = f"{GR}●{R}" if count >= MIN_CAPTURE else f"{YL}●{R}"
            print(f"  {dot}  {CY}{sid:<10}{R} {WH}{sname:<20}{R} {DM}{count} images{R}")
    print()


# ══════════════════════════════════════════════════════════════
#  Face capture
# ══════════════════════════════════════════════════════════════

def capture_faces(name: str, sid: str, num: int = TARGET_CAPTURE) -> bool:
    """
    Open the webcam and capture *num* face images for the given student.

    Images are saved as JPEG files inside:
        dataset/<sid>_<name>/

    Returns:
        True  — if at least MIN_CAPTURE images were captured successfully.
        False — if capture was aborted or too few faces were detected.
    """
    folder = os.path.join(DATASET_DIR, f"{sid}_{name}")
    os.makedirs(folder, exist_ok=True)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        err("Cannot open webcam. Check that it is connected and not in use.")
        return False

    count = 0
    info(f"Capturing for {B}{name}{R} — slowly move your head for variety.")
    info("Press  Q  to cancel at any time.\n")
    time.sleep(0.8)

    while count < num:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            count += 1
            face   = cv2.resize(frame[y:y+h, x:x+w], FACE_SIZE)
            cv2.imwrite(os.path.join(folder, f"{sid}_{count:03d}.jpg"), face)

            # Styled bounding box overlay
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 220, 150), 2)
            cv2.rectangle(frame, (x, y-38), (x+w, y), (0, 220, 150), -1)
            cv2.putText(frame, f"{name}  {count}/{num}",
                        (x+6, y-12), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)

        # Progress fill bar at the bottom of the video frame
        prog_w = int(frame.shape[1] * count / num)
        cv2.rectangle(frame, (0, frame.shape[0]-8),
                      (frame.shape[1], frame.shape[0]), (40, 40, 40), -1)
        cv2.rectangle(frame, (0, frame.shape[0]-8),
                      (prog_w, frame.shape[0]), (0, 220, 150), -1)

        # Corner watermark
        cv2.putText(frame, "SMART ATTENDANCE  |  REGISTRATION",
                    (10, frame.shape[0]-18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

        cv2.imshow(f"Registering: {name}", frame)
        progress_bar(count, num)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print()  # newline after inline progress bar

    if count >= MIN_CAPTURE:
        ok(f"Captured {count} images  →  {DM}{folder}/{R}")
        return True

    err(f"Only {count} image(s) captured — need at least {MIN_CAPTURE}. Please try again.")
    return False


# ══════════════════════════════════════════════════════════════
#  Registration flow
# ══════════════════════════════════════════════════════════════

def register_flow() -> None:
    """Run one complete student registration cycle."""
    banner()
    students = load_students()
    show_registered(students)

    step(1, "Enter Student Details")
    print()
    sid  = input(f"  {CY}Student ID   {R}(e.g. S001) : ").strip().upper()
    name = input(f"  {CY}Student Name {R}(e.g. Alice): ").strip().title()

    if not sid or not name:
        err("Student ID and Name cannot be empty.")
        return

    if sid in students:
        warn(f"'{sid}' is already registered as '{students[sid]}'.")
        choice = input(f"  {YL}Re-register? [y/N]{R}: ").strip().lower()
        if choice != 'y':
            info("Skipping — keeping the existing registration.")
            return
        old_folder = os.path.join(DATASET_DIR, f"{sid}_{students[sid]}")
        if os.path.exists(old_folder):
            shutil.rmtree(old_folder)
            ok("Old images removed.")

    step(2, "Webcam Face Capture")
    success = capture_faces(name, sid, num=TARGET_CAPTURE)

    if success:
        students[sid] = name
        save_students(students)
        print(f"\n  {BG_GR}{WH} SUCCESS {R} {B}{name}{R} (ID: {CY}{sid}{R}) registered!")
        print(f"\n  {DM}Next step →  python module2_encode_faces.py{R}\n")
    else:
        print(f"\n  {RD}{B}Registration incomplete. Please try again.{R}\n")


def main() -> None:
    """Entry point — loop registration until the user exits."""
    while True:
        register_flow()
        again = input(f"  {CY}Register another student? [y/N]{R}: ").strip().lower()
        if again != 'y':
            break

    banner()
    students = load_students()
    show_registered(students)
    print(f"  {GR}{B}All registrations done!{R}")
    print(f"  {DM}Run: python module2_encode_faces.py{R}\n")


if __name__ == "__main__":
    main()