# ============================================================
#  MAIN — Smart Attendance System
#  Menu-driven launcher for all 5 modules
# ============================================================
"""
Run this file to access all modules from a single menu.

    python main.py

Modules:
    1 — Face Registration       (module1_register_face.py)
    2 — Face Encoding           (module2_encode_faces.py)
    3 — Train Model             (module3_train_model.py)
    4 — Take Attendance (Auto)  (module4_take_attendance.py)
    5 — Reports & Visualization (module5_visualization.py)
    0 — Exit
"""

import os
import sys

# ── ANSI Color Palette ────────────────────────────────────────
R     = "\033[0m"    # Reset
B     = "\033[1m"    # Bold
DM    = "\033[2m"    # Dim
CY    = "\033[96m"   # Cyan
GR    = "\033[92m"   # Green
YL    = "\033[93m"   # Yellow
RD    = "\033[91m"   # Red
BL    = "\033[94m"   # Blue
WH    = "\033[97m"   # White
BG_BL = "\033[44m"   # Blue background
BG_GR = "\033[42m"   # Green background
BG_CY = "\033[46m"   # Cyan background


# ══════════════════════════════════════════════════════════════
#  Terminal helpers
# ══════════════════════════════════════════════════════════════

def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def banner() -> None:
    """Print the main menu banner."""
    clear()
    print(f"\n{CY}{B}{'═'*60}{R}")
    print(f"{CY}{B}   ██████╗ ███████╗ ██████╗{R}")
    print(f"{CY}{B}   ██╔══██╗██╔════╝██╔════╝{R}")
    print(f"{CY}{B}   ██████╔╝█████╗  ██║     {R}")
    print(f"{CY}{B}   ██╔══██╗██╔══╝  ██║     {R}")
    print(f"{CY}{B}   ██║  ██║███████╗╚██████╗{R}")
    print(f"{CY}{B}{'─'*60}{R}")
    print(f"{CY}{B}      Smart Attendance System  —  Main Menu{R}")
    print(f"{CY}{B}{'═'*60}{R}\n")


def check_file(path: str) -> str:
    """Return a colored tick or cross depending on whether *path* exists."""
    return f"{GR}✔{R}" if os.path.exists(path) else f"{RD}✖{R}"


def show_menu() -> None:
    """Display the interactive menu with prerequisite status indicators."""
    banner()

    # Prerequisite file status
    students_ok  = check_file("students.json")
    encodings_ok = check_file("encodings.pkl")
    model_ok     = check_file("trained_model.pkl")
    csv_ok       = check_file("attendance.csv")

    print(f"  {DM}System Status:{R}")
    print(f"    {students_ok}  students.json      {encodings_ok}  encodings.pkl")
    print(f"    {model_ok}  trained_model.pkl  {csv_ok}  attendance.csv\n")
    print(f"  {DM}{'─'*54}{R}")

    menu_items = [
        ("1", "Register Student Faces",         "Capture face images via webcam",           CY),
        ("2", "Encode Faces",                   "Generate 128-d face descriptors",           BL),
        ("3", "Train Recognition Model",        "Train KNN / SVM on face encodings",         YL),
        ("4", "Take Attendance",        "Camera auto-identifies & marks students",   GR),
        ("5", "View Reports & Visualizations",  "Charts, heatmap, dashboard, text report",   WH),
        ("0", "Exit",                           "",                                           RD),
    ]

    for key, title, desc, color in menu_items:
        if key == "0":
            print(f"  {DM}{'─'*54}{R}")
        bullet = f"{BG_BL}{WH} {key} {R}" if key != "0" else f"{BG_BL}{WH} {key} {R}"
        desc_str = f"  {DM}{desc}{R}" if desc else ""
        print(f"  {bullet}  {color}{B}{title}{R}{desc_str}")

    print(f"\n  {DM}{'─'*54}{R}")


def warn(msg: str) -> None:
    print(f"\n  {YL}⚠{R}  {msg}")


def info(msg: str) -> None:
    print(f"\n  {CY}ℹ{R}  {msg}")


# ══════════════════════════════════════════════════════════════
#  Prerequisite guards
# ══════════════════════════════════════════════════════════════

def require(files: list) -> bool:
    """
    Check that all required files exist before running a module.
    Prints a helpful error and returns False if any are missing.
    """
    missing = [f for f, _ in files if not os.path.exists(f)]
    if missing:
        print(f"\n  {RD}✖  Cannot run this module yet.{R}")
        for fpath, hint in files:
            if not os.path.exists(fpath):
                print(f"     {RD}Missing:{R} {fpath}  {DM}→ {hint}{R}")
        input(f"\n  Press Enter to return to the menu! ")
        return False
    return True


# ══════════════════════════════════════════════════════════════
#  Module launchers
# ══════════════════════════════════════════════════════════════

def run_module1() -> None:
    """Launch Module 1 — Face Registration."""
    from module1_register_face import main as m1_main
    m1_main()
    input(f"\n  {DM}Press Enter to return to the main menu! {R}")


def run_module2() -> None:
    """Launch Module 2 — Face Encoding."""
    ok = require([
        ("students.json", "Run Module 1 first to register students"),
        ("dataset",       "Run Module 1 first to capture face images"),
    ])
    if not ok:
        return
    from module2_encode_faces import encode_all
    encode_all()
    input(f"\n  {DM}Press Enter to return to the main menu! {R}")


def run_module3() -> None:
    """Launch Module 3 — Model Training."""
    ok = require([
        ("encodings.pkl", "Run Module 2 first to generate face encodings"),
    ])
    if not ok:
        return
    from module3_train_model import train_and_test
    train_and_test()
    input(f"\n  {DM}Press Enter to return to the main menu! {R}")


def run_module4() -> None:
    """Launch Module 4 — Auto Attendance."""
    ok = require([
        ("students.json",    "Run Module 1 first"),
        ("encodings.pkl",    "Run Module 2 first"),
        ("trained_model.pkl","Run Module 3 first"),
    ])
    if not ok:
        return
    from module4_take_attendance import run
    run()
    input(f"\n  {DM}Press Enter to return to the main menu! {R}")


def run_module5() -> None:
    """Launch Module 5 — Reports & Visualization."""
    ok = require([
        ("attendance.csv", "Run Module 4 first to record attendance"),
    ])
    if not ok:
        return
    from module5_visualization import (
        banner, load_data, plot_daily, plot_student_percent,
        plot_pie, plot_heatmap, plot_dashboard,
        generate_text_report, print_terminal_report, section
    )
    import module5_visualization as m5

    m5.banner()

    df, all_names = load_data()
    info_str = (
        f"Loaded {len(df)} record(s)  |  "
        f"{df['Date'].nunique()} day(s)  |  "
        f"{len(all_names)} student(s)"
    )
    print(f"  {CY}ℹ{R}  {info_str}\n")

    section("Generating Charts")
    plot_daily(df)
    pct_df = plot_student_percent(df, all_names)
    plot_pie(df, all_names)
    plot_heatmap(df, all_names)
    plot_dashboard(df, pct_df, all_names)

    section("Generating Text Report")
    generate_text_report(df, pct_df, all_names)
    print_terminal_report(df, pct_df, all_names)

    print(f"\n  {BG_GR}{WH} ALL DONE {R}  Reports saved in {CY}reports/{R}\n")
    input(f"\n  {DM}Press Enter to return to the main menu! {R}")


# ══════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════

DISPATCH = {
    "1": run_module1,
    "2": run_module2,
    "3": run_module3,
    "4": run_module4,
    "5": run_module5,
}


def main() -> None:
    """Main menu loop — runs until the user selects 0 to exit."""
    while True:
        show_menu()
        choice = input(f"\n  {CY}Enter your choice{R} [0-5]: ").strip()

        if choice == "0":
            clear()
            print(f"\n  {GR}{B}Goodbye! Exiting Smart Attendance System.{R}\n")
            sys.exit(0)

        handler = DISPATCH.get(choice)
        if handler:
            handler()
        else:
            warn(f"'{choice}' is not a valid option. Please choose 0–5.")
            input(f"  Press Enter to continue …")


if __name__ == "__main__":
    main()