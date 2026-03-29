# ============================================================
#  MODULE 4 — LIVE ATTENDANCE  (PREDICTION MODULE)
#  Smart Attendance System using Face Recognition
# ============================================================
"""
Runs an automatic attendance session where the webcam identifies
any registered student by their face — no name or ID input required.

Flow:
    1. Webcam opens automatically.
    2. System scans faces against ALL registered student encodings.
    3. When a match is found → PRESENT is marked in attendance.csv.
    4. The student's name is shown on screen with a green box.
    5. Already-marked students are skipped automatically.
    6. Press  Q  to end the session.

Run:
    python module4_take_attendance.py

Prerequisites:
    python module1_register_face.py
    python module2_encode_faces.py
    python module3_train_model.py

Next step:
    python module5_visualization.py
"""

import csv
import json
import os
import pickle
import sys
import time
from datetime import datetime

import cv2
import face_recognition
import numpy as np

# ── File Constants ────────────────────────────────────────────
ENCODINGS_FILE  = "encodings.pkl"
MODEL_FILE      = "trained_model.pkl"
STUDENTS_FILE   = "students.json"
ATTENDANCE_FILE = "attendance.csv"

# ── Recognition Tuning ────────────────────────────────────────
TOLERANCE      = 0.52   # lower → stricter face matching
FRAME_SKIP     = 2      # process face detection every N frames (performance)
WEBCAM_WIDTH   = 1280
WEBCAM_HEIGHT  = 720
CONFIRM_FRAMES = 5      # consecutive matching frames needed before marking

# ── ANSI Colors ───────────────────────────────────────────────
R      = "\033[0m"   # Reset
B      = "\033[1m"   # Bold
DM     = "\033[2m"   # Dim
CY     = "\033[96m"  # Cyan
GR     = "\033[92m"  # Green
YL     = "\033[93m"  # Yellow
RD     = "\033[91m"  # Red
BL     = "\033[94m"  # Blue
WH     = "\033[97m"  # White
BG_GR  = "\033[42m"  # Green background
BG_RD  = "\033[41m"  # Red background
BG_BL  = "\033[44m"  # Blue background
BG_YL  = "\033[43m"  # Yellow background


# ══════════════════════════════════════════════════════════════
#  Terminal UI helpers
# ══════════════════════════════════════════════════════════════

def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def ok(msg: str)   -> None: print(f"  {GR}✔{R}  {msg}")
def warn(msg: str) -> None: print(f"  {YL}⚠{R}  {msg}")
def err(msg: str)  -> None: print(f"  {RD}✖{R}  {msg}")
def info(msg: str) -> None: print(f"  {CY}ℹ{R}  {msg}")


def banner(today: str) -> None:
    """Print the module header banner with today's date."""
    clear()
    print(f"\n{GR}{B}{'═'*60}{R}")
    print(f"{GR}{B}   Smart Attendance System  |  {today}{R}")
    print(f"{GR}{B}{'═'*60}{R}")
    print(f"  {DM}   MODULE 4 — Auto Face Recognition & Attendance{R}\n")


def section(title: str) -> None:
    print(f"\n  {BG_BL}{WH}  {title}  {R}\n")


# ══════════════════════════════════════════════════════════════
#  CSV helpers
# ══════════════════════════════════════════════════════════════

CSV_FIELDNAMES = ["Student ID", "Student Name", "Date", "Time", "Status"]


def init_csv() -> None:
    """Create attendance.csv with headers if it does not already exist."""
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(CSV_FIELDNAMES)


def already_marked_today(sid: str, today: str) -> tuple:
    """
    Check whether a student has already been marked Present today.

    Returns:
        (True,  time_string)  if already marked.
        (False, None)         otherwise.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        return False, None
    with open(ATTENDANCE_FILE) as f:
        for row in csv.DictReader(f):
            if (row["Student ID"] == sid
                    and row["Date"] == today
                    and row["Status"] == "Present"):
                return True, row["Time"]
    return False, None


def get_all_marked_today(today: str) -> dict:
    """Return a mapping {student_id: time_string} for all Present records today."""
    marked: dict = {}
    if not os.path.exists(ATTENDANCE_FILE):
        return marked
    with open(ATTENDANCE_FILE) as f:
        for row in csv.DictReader(f):
            if row["Date"] == today and row["Status"] == "Present":
                marked[row["Student ID"]] = row["Time"]
    return marked


def write_record(sid: str, sname: str, date: str, t: str, status: str) -> None:
    """Append a single attendance record to attendance.csv."""
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([sid, sname, date, t, status])


def clear_today_attendance(today: str) -> None:
    """
    Remove all rows for *today* from attendance.csv.
    Records from previous days are preserved.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        return
    with open(ATTENDANCE_FILE, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["Date"] != today]
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept)


# ══════════════════════════════════════════════════════════════
#  Attendance board
# ══════════════════════════════════════════════════════════════

def show_board(students: dict, marked: dict, today: str) -> None:
    """Display a live attendance status table for the current session."""
    section("Today's Attendance Board")
    print(f"  {DM}  {'ID':<8} {'Name':<22} {'Status':<16} {'Time'}{R}")
    print(f"  {DM}  {'─'*6} {'─'*20} {'─'*14} {'─'*8}{R}")

    present_count = 0
    for sid, sname in students.items():
        if sid in marked:
            present_count += 1
            dot   = f"{GR}●{R}"
            label = f"{BG_GR}{WH} PRESENT {R}"
            tstr  = f"{GR}{marked[sid]}{R}"
        else:
            dot   = f"{YL}○{R}"
            label = f"{DM} waiting  {R}"
            tstr  = f"{DM}──{R}"
        print(f"  {dot}  {CY}{sid:<8}{R} {WH}{sname:<22}{R} {label}  {tstr}")

    absent_count = len(students) - present_count
    print(
        f"\n  {GR}{B}Present: {present_count}{R}   "
        f"{RD}{B}Absent: {absent_count}{R}   "
        f"{DM}Total: {len(students)}{R}\n"
    )


# ══════════════════════════════════════════════════════════════
#  Build a fast lookup: encoding → (sid, name)
# ══════════════════════════════════════════════════════════════

def build_lookup(enc_data: dict) -> tuple:
    """
    Flatten encodings.pkl into parallel lists for fast comparison.

    Returns:
        (all_encodings, all_sids, all_names)
    """
    return (
        enc_data["encodings"],
        enc_data["ids"],
        enc_data["names"],
    )


# ══════════════════════════════════════════════════════════════
#  Auto attendance session  (camera stays open the whole time)
# ══════════════════════════════════════════════════════════════

def run_auto_session(
    students: dict,
    enc_data: dict,
    marked: dict,
    today: str,
) -> dict:
    """
    Open the webcam and continuously recognize faces.

    For every frame (sampled at FRAME_SKIP), compare detected faces
    against ALL registered encodings.  When a face has been confirmed
    for CONFIRM_FRAMES consecutive frames, mark that student present.

    Press  Q  or  Esc  to end the session early.

    Returns the updated *marked* dict.
    """
    all_encodings, all_sids, all_names = build_lookup(enc_data)

    if not all_encodings:
        err("No encodings found. Please re-run Module 2.")
        return marked

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        err("Cannot open webcam. Check it is connected and not in use.")
        return marked

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    # Confirmation counter per student (avoids single-frame false positives)
    confirm_counts: dict = {}   # sid → int

    frame_count  = 0
    recent_flash = {}           # sid → timestamp, for on-screen "MARKED" flash

    info("Camera is open. Show your face to mark attendance.")
    info("Press  Q  or  Esc  to end the session.\n")
    time.sleep(0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        h, w = frame.shape[:2]

        # ── Header bar ───────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 58), (13, 17, 23), -1)
        cv2.putText(
            frame, "SMART ATTENDANCE  |  AUTO FACE RECOGNITION",
            (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.60, (88, 166, 255), 1,
        )

        # Status line: how many marked so far
        status_txt = (
            f"Present: {len(marked)}/{len(students)}  |  "
            f"Press Q to finish session"
        )
        cv2.putText(
            frame, status_txt,
            (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (201, 209, 217), 1,
        )

        # ── Face processing (every FRAME_SKIP frames) ────────
        if frame_count % FRAME_SKIP == 0:
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs  = face_recognition.face_locations(rgb)
            encs  = face_recognition.face_encodings(rgb, locs)

            for (top, right, bottom, left), enc in zip(locs, encs):
                # Scale back to full resolution
                top, right, bottom, left = top*2, right*2, bottom*2, left*2

                # Compare against all registered encodings
                dists    = face_recognition.face_distance(all_encodings, enc)
                best_idx = int(np.argmin(dists))
                min_dist = float(dists[best_idx])
                conf     = round(1 - min_dist, 2)
                match    = min_dist <= TOLERANCE

                if match:
                    sid   = all_sids[best_idx]
                    sname = all_names[best_idx]

                    already, _ = already_marked_today(sid, today)

                    if already or sid in marked:
                        # Already marked — show blue box
                        box_col   = (255, 191, 0)
                        label_txt = f"{sname}  ✔ ALREADY MARKED"
                        sub_txt   = "Attendance recorded earlier"
                        confirm_counts.pop(sid, None)
                    else:
                        # Increment confirmation counter
                        confirm_counts[sid] = confirm_counts.get(sid, 0) + 1
                        box_col   = (63, 185, 80)
                        left_frames = CONFIRM_FRAMES - confirm_counts[sid]
                        label_txt = (
                            f"{sname}  {conf:.0%}"
                            if left_frames > 0
                            else f"{sname}  ✔ CONFIRMED"
                        )
                        sub_txt = (
                            f"Hold still… ({confirm_counts[sid]}/{CONFIRM_FRAMES})"
                            if left_frames > 0
                            else "Marking attendance…"
                        )

                        # Enough consecutive frames → mark present
                        if confirm_counts[sid] >= CONFIRM_FRAMES:
                            now_time = datetime.now().strftime("%H:%M:%S")
                            write_record(sid, sname, today, now_time, "Present")
                            marked[sid] = now_time
                            recent_flash[sid] = time.time()
                            confirm_counts.pop(sid, None)
                            print(f"  {BG_GR}{WH} MARKED {R}  {B}{sname}{R} "
                                  f"({CY}{sid}{R})  →  {GR}PRESENT{R}  at {now_time}")
                else:
                    box_col   = (247, 129, 102)
                    label_txt = f"Unknown  {conf:.0%}"
                    sub_txt   = "Face not recognized"
                    sid = None

                # ── Bounding box with corner accents ─────────
                cv2.rectangle(frame, (left, top), (right, bottom), box_col, 2)
                L = 18
                for (cx, cy, dx, dy) in [
                    (left,  top,    1,  1),
                    (right, top,   -1,  1),
                    (left,  bottom, 1, -1),
                    (right, bottom,-1, -1),
                ]:
                    cv2.line(frame, (cx, cy), (cx + dx*L, cy),  box_col, 3)
                    cv2.line(frame, (cx, cy), (cx, cy + dy*L),  box_col, 3)

                # Label below the bounding box
                cv2.rectangle(
                    frame, (left, bottom + 2), (right, bottom + 46),
                    box_col, -1,
                )
                cv2.putText(
                    frame, label_txt,
                    (left + 6, bottom + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1,
                )
                cv2.putText(
                    frame, sub_txt,
                    (left + 6, bottom + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1,
                )

        # ── Flash "MARKED!" overlay for 2 s after marking ────
        now = time.time()
        y_offset = 80
        for sid, flash_time in list(recent_flash.items()):
            if now - flash_time < 2.0:
                sname = students.get(sid, sid)
                cv2.rectangle(frame, (w//2 - 220, y_offset - 28),
                              (w//2 + 220, y_offset + 8), (63, 185, 80), -1)
                cv2.putText(
                    frame, f"✔  {sname}  MARKED PRESENT",
                    (w//2 - 210, y_offset),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2,
                )
                y_offset += 46
            else:
                del recent_flash[sid]

        cv2.imshow("Smart Attendance — Auto Recognition  [Q to finish]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or Esc
            break

        # Auto-close when all students are marked
        if len(marked) == len(students):
            info("All students marked! Closing camera in 2 seconds…")
            time.sleep(2)
            break

    cap.release()
    cv2.destroyAllWindows()
    return marked


# ══════════════════════════════════════════════════════════════
#  End-of-session summary
# ══════════════════════════════════════════════════════════════

def close_session(students: dict, marked: dict, today: str) -> None:
    """
    Mark all absent students in the CSV and print the session summary.
    Called once when the operator presses Q to end the session.
    """
    end_time = datetime.now().strftime("%H:%M:%S")
    absent:  list = []

    for sid, sname in students.items():
        if sid not in marked:
            write_record(sid, sname, today, end_time, "Absent")
            absent.append((sid, sname))

    clear()
    today_fmt = datetime.now().strftime("%A, %d %B %Y")
    print(f"\n{GR}{B}{'═'*60}{R}")
    print(f"{GR}{B}   Smart Attendance System{R}")
    print(f"{GR}{B}{'═'*60}{R}\n")
    print(f"  {BG_BL}{WH}  SESSION COMPLETE — {today_fmt}  {R}\n")

    print(f"  {GR}{B}Present ({len(marked)}){R}")
    for sid in marked:
        sname = students.get(sid, sid)
        print(f"     {GR}●{R}  {WH}{sname}{R}  {DM}({sid})  at {marked[sid]}{R}")

    print(f"\n  {RD}{B}Absent ({len(absent)}){R}")
    if absent:
        for sid, sname in absent:
            print(f"     {RD}○{R}  {WH}{sname}{R}  {DM}({sid}){R}")
    else:
        print(f"     {GR}All students are present!{R}")

    print(f"\n  {DM}Full log saved  →  {ATTENDANCE_FILE}{R}")
    print(f"  {DM}Next step:  python module5_visualization.py{R}\n")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def run() -> None:
    """Validate prerequisites, then start the auto attendance session."""
    for fpath, label in [
        (ENCODINGS_FILE, "Module 2"),
        (MODEL_FILE,     "Module 3"),
        (STUDENTS_FILE,  "Module 1"),
    ]:
        if not os.path.exists(fpath):
            err(f"{fpath} not found — run {label} first.")
            sys.exit(1)

    with open(ENCODINGS_FILE, "rb") as f:
        enc_data = pickle.load(f)
    with open(STUDENTS_FILE) as f:
        students = json.load(f)

    init_csv()
    today  = datetime.now().strftime("%Y-%m-%d")
    marked = get_all_marked_today(today)

    banner(today)
    info(f"Session started for {B}{today}{R}.")
    info(f"Registered students: {', '.join(f'{CY}{n}{R}' for n in students.values())}")

    # ── Offer reset if today already has records ──────────────
    if marked:
        print(f"\n  {BG_YL}{WH}  WARNING  {R}  "
              f"{YL}{len(marked)} student(s) already marked today:{R}")
        for sid in marked:
            print(f"     {GR}●{R}  {WH}{students.get(sid, sid)}{R}  "
                  f"{DM}at {marked[sid]}{R}")
        print(f"\n  {WH}Do you want to clear today's attendance and start fresh?{R}")
        print(f"  {DM}(This will NOT affect records from previous days.){R}\n")
        choice = input(f"  Clear and restart? [{YL}y{R}/N]: ").strip().lower()
        if choice == 'y':
            clear_today_attendance(today)
            marked = {}
            ok("Today's attendance cleared. Starting fresh.\n")
        else:
            info("Keeping existing records. Already-marked students will be skipped.\n")

    # ── Show attendance board before starting ─────────────────
    show_board(students, marked, today)

    section("How It Works")
    print(f"  {WH}The camera will open and automatically identify each student.{R}\n")
    print(f"  {CY}Step 1{R}  Stand in front of the camera and look straight at it.")
    print(f"  {CY}Step 2{R}  Hold still for a moment — your face will be matched.")
    print(f"  {CY}Step 3{R}  When matched, attendance is marked {GR}PRESENT{R} automatically.")
    print(f"  {CY}Step 4{R}  The next student steps up and repeats.")
    print(f"\n  {YL}Press  Q  or  Esc  to end the session at any time.{R}\n")
    input("  Press Enter to open the camera …")

    # ── Run the auto recognition session ─────────────────────
    marked = run_auto_session(students, enc_data, marked, today)

    close_session(students, marked, today)


if __name__ == "__main__":
    run()