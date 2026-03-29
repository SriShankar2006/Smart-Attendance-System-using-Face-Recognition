# ============================================================
#  MODULE 5 — REPORTING & VISUALIZATION
#  Smart Attendance System using Face Recognition
# ============================================================
"""
Generates charts, a summary dashboard, and a plain-text report from
the attendance records produced by Module 4.

Outputs (saved to reports/):
    1_daily_attendance.png    — bar chart of daily attendance counts
    2_student_percentage.png  — horizontal bar chart of per-student %
    3_pie_latest_day.png      — present/absent pie for the latest session
    4_heatmap.png             — P/A heatmap across all days and students
    5_dashboard.png           — summary dashboard with KPI table
    attendance_report.txt     — plain-text summary report

Run:
    python module5_visualization.py

Prerequisites:
    python module4_take_attendance.py   (attendance.csv must exist)
"""

import json
import os
import sys
from datetime import datetime

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ── File / Directory Constants ────────────────────────────────
ATTENDANCE_FILE = "attendance.csv"
STUDENTS_FILE   = "students.json"
REPORT_DIR      = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# ── Attendance threshold ──────────────────────────────────────
ELIGIBILITY_THRESHOLD = 75  # % required to be considered eligible

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

# ── Dark chart theme (GitHub-style) ──────────────────────────
DARK_BG   = "#0d1117"
DARK_SURF = "#161b22"
DARK_BORD = "#30363d"
COL_TEXT  = "#c9d1d9"
COL_DIM   = "#8b949e"
COL_GR    = "#3fb950"
COL_RD    = "#f85149"
COL_BL    = "#58a6ff"
COL_YL    = "#e3b341"
COL_MG    = "#bc8cff"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   DARK_SURF,
    "axes.edgecolor":   DARK_BORD,
    "axes.labelcolor":  COL_DIM,
    "xtick.color":      COL_DIM,
    "ytick.color":      COL_DIM,
    "text.color":       COL_TEXT,
    "grid.color":       DARK_BORD,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})


# ══════════════════════════════════════════════════════════════
#  Terminal UI helpers
# ══════════════════════════════════════════════════════════════

def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def ok(msg: str)   -> None: print(f"  {GR}✔{R}  {msg}")
def info(msg: str) -> None: print(f"  {CY}ℹ{R}  {msg}")
def err(msg: str)  -> None: print(f"  {RD}✖{R}  {msg}")


def section(title: str) -> None:
    print(f"\n  {BG_BL}{WH}  {title}  {R}\n")


def banner() -> None:
    """Print the module header banner."""
    clear()
    print(f"\n{YL}{B}{'═'*58}{R}")
    print(f"{YL}{B}   Smart Attendance System{R}")
    print(f"{YL}{B}   MODULE 5 — Reports & Visualization{R}")
    print(f"{YL}{B}{'═'*58}{R}\n")


# ══════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, list]:
    """
    Load attendance.csv and the students registry.

    Returns:
        df        — DataFrame with attendance records (Date column as datetime).
        all_names — ordered list of all student names.
    """
    if not os.path.exists(ATTENDANCE_FILE):
        err(f"'{ATTENDANCE_FILE}' not found — run Module 4 first.")
        sys.exit(1)

    df = pd.read_csv(ATTENDANCE_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    students: dict = {}
    if os.path.exists(STUDENTS_FILE):
        with open(STUDENTS_FILE) as f:
            students = json.load(f)

    all_names = (
        list(students.values())
        or df["Student Name"].unique().tolist()
    )
    return df, all_names


# ══════════════════════════════════════════════════════════════
#  Chart helpers
# ══════════════════════════════════════════════════════════════

def _style_ax(
    ax,
    title:  str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """Apply consistent dark-theme styling to a matplotlib Axes object."""
    ax.set_facecolor(DARK_SURF)
    for spine in ax.spines.values():
        spine.set_color(DARK_BORD)
    ax.set_title(title, color=COL_TEXT, fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=COL_DIM, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=COL_DIM, fontsize=9)
    ax.tick_params(colors=COL_DIM, labelsize=8)
    ax.grid(True, axis="y", color=DARK_BORD, linestyle="--", alpha=0.4)


def _save(path: str) -> None:
    """Save the current figure to *path* with the dark background."""
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)


# ══════════════════════════════════════════════════════════════
#  Individual charts
# ══════════════════════════════════════════════════════════════

def plot_daily(df: pd.DataFrame) -> None:
    """Bar chart showing how many students were present on each date."""
    daily = (
        df[df["Status"] == "Present"]
        .groupby(df["Date"].dt.strftime("%d %b"))["Student Name"]
        .count()
    )

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=DARK_BG)
    bars = ax.bar(daily.index, daily.values, color=COL_BL,
                  edgecolor=DARK_BORD, width=0.55, zorder=3)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            str(int(bar.get_height())),
            ha="center", va="bottom",
            color=COL_TEXT, fontsize=10, fontweight="bold",
        )

    _style_ax(ax, "Daily Attendance Count", "Date", "Students Present")
    ax.tick_params(axis="x", rotation=30)
    ax.set_facecolor(DARK_SURF)
    plt.tight_layout()

    path = os.path.join(REPORT_DIR, "1_daily_attendance.png")
    _save(path)
    plt.show()
    ok("1_daily_attendance.png")


def plot_student_percent(df: pd.DataFrame, all_names: list) -> pd.DataFrame:
    """
    Horizontal bar chart of each student's overall attendance percentage.

    Returns the computed percentage DataFrame for use in other charts.
    """
    total = df["Date"].nunique()
    rows  = []
    for name in all_names:
        days = (
            df[(df["Student Name"] == name) & (df["Status"] == "Present")]
            ["Date"].nunique()
        )
        pct = round(days / total * 100, 1) if total else 0.0
        rows.append({"Name": name, "Days": days, "Pct": pct})

    pct_df = pd.DataFrame(rows).sort_values("Pct")
    colors = [
        COL_GR if p >= ELIGIBILITY_THRESHOLD else COL_RD
        for p in pct_df["Pct"]
    ]

    fig, ax = plt.subplots(
        figsize=(9, max(4, len(all_names))), facecolor=DARK_BG
    )
    bars = ax.barh(
        pct_df["Name"], pct_df["Pct"],
        color=colors, edgecolor=DARK_BORD, height=0.55, zorder=3,
    )
    ax.axvline(
        ELIGIBILITY_THRESHOLD, color=COL_YL, linestyle="--",
        linewidth=1.5, label=f"{ELIGIBILITY_THRESHOLD}% threshold", zorder=4,
    )

    for bar, pct in zip(bars, pct_df["Pct"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct}%",
            va="center", color=COL_TEXT, fontsize=9, fontweight="bold",
        )

    _style_ax(
        ax,
        f"Student Attendance %  (out of {total} day(s))",
        "Attendance (%)", "",
    )
    ax.set_xlim(0, 115)
    ax.legend(
        facecolor=DARK_SURF, edgecolor=DARK_BORD,
        labelcolor=COL_TEXT, fontsize=8,
    )
    plt.tight_layout()

    path = os.path.join(REPORT_DIR, "2_student_percentage.png")
    _save(path)
    plt.show()
    ok("2_student_percentage.png")
    return pct_df


def plot_pie(df: pd.DataFrame, all_names: list) -> None:
    """Pie chart of present vs. absent for the most recent session."""
    latest  = df["Date"].max()
    present = (
        df[(df["Date"] == latest) & (df["Status"] == "Present")]
        ["Student Name"].tolist()
    )
    absent  = [n for n in all_names if n not in present]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=DARK_BG)
    sizes  = [len(present), len(absent)]
    colors = [COL_GR, COL_RD]
    labels = [f"Present  ({len(present)})", f"Absent  ({len(absent)})"]

    _, _, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=140, explode=(0.04, 0.04),
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        textprops={"color": COL_TEXT, "fontsize": 10},
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontweight("bold")

    ax.set_title(
        f"Attendance — {latest.strftime('%d %b %Y')}",
        color=COL_TEXT, fontsize=12, fontweight="bold", pad=14,
    )
    plt.tight_layout()

    path = os.path.join(REPORT_DIR, "3_pie_latest_day.png")
    _save(path)
    plt.show()
    ok("3_pie_latest_day.png")


def plot_heatmap(df: pd.DataFrame, all_names: list) -> None:
    """Heatmap grid showing P (Present) / A (Absent) for every student-day."""
    dates    = sorted(df["Date"].dt.date.unique())
    col_lbls = [str(d) for d in dates]

    heat_num = pd.DataFrame(0,  index=all_names, columns=col_lbls)
    heat_ann = pd.DataFrame("", index=all_names, columns=col_lbls)

    for _, row in df.iterrows():
        name = row["Student Name"]
        date = str(row["Date"].date())
        if name in heat_num.index and date in heat_num.columns:
            v = 1 if row["Status"] == "Present" else 0
            heat_num.loc[name, date] = v
            heat_ann.loc[name, date] = "P" if v else "A"

    fig, ax = plt.subplots(
        figsize=(max(8, len(dates) * 1.5), max(4, len(all_names) * 0.9)),
        facecolor=DARK_BG,
    )
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    sns.heatmap(
        heat_num.astype(int), annot=heat_ann, fmt="",
        cmap=cmap, ax=ax, linewidths=1.5, linecolor=DARK_BG,
        cbar=False, vmin=0, vmax=1,
        xticklabels=[d[5:] for d in col_lbls],
        annot_kws={"size": 11, "weight": "bold", "color": DARK_BG},
    )
    ax.set_title(
        "Attendance Heatmap  (P = Present  |  A = Absent)",
        color=COL_TEXT, fontsize=12, fontweight="bold", pad=12,
    )
    ax.set_ylabel("Student",       color=COL_DIM, fontsize=9)
    ax.set_xlabel("Date  (MM-DD)", color=COL_DIM, fontsize=9)
    ax.tick_params(colors=COL_DIM)
    for spine in ax.spines.values():
        spine.set_color(DARK_BORD)
    plt.tight_layout()

    path = os.path.join(REPORT_DIR, "4_heatmap.png")
    _save(path)
    plt.show()
    ok("4_heatmap.png")


def plot_dashboard(
    df: pd.DataFrame,
    pct_df: pd.DataFrame,
    all_names: list,
) -> None:
    """Multi-panel summary dashboard combining all key metrics."""
    total_days     = df["Date"].nunique()
    avg_pct        = pct_df["Pct"].mean() if len(pct_df) else 0.0
    above          = int((pct_df["Pct"] >= ELIGIBILITY_THRESHOLD).sum())
    below          = len(pct_df) - above
    latest_day     = df["Date"].max().strftime("%d %b %Y")
    latest_present = int(
        df[(df["Date"] == df["Date"].max()) & (df["Status"] == "Present")]
        ["Student Name"].nunique()
    )

    fig = plt.figure(figsize=(16, 9), facecolor=DARK_BG)
    fig.suptitle(
        "Smart Attendance System — Summary Dashboard",
        color=COL_TEXT, fontsize=15, fontweight="bold", y=1.01,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40)

    # ── Panel 1: Daily attendance bar ─────────────────────────
    ax1  = fig.add_subplot(gs[0, :2])
    daily = (
        df[df["Status"] == "Present"]
        .groupby(df["Date"].dt.strftime("%d %b"))["Student Name"]
        .count()
    )
    bars = ax1.bar(daily.index, daily.values, color=COL_BL,
                   edgecolor=DARK_BORD, width=0.5, zorder=3)
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            str(int(bar.get_height())),
            ha="center", color=COL_TEXT, fontsize=9, fontweight="bold",
        )
    _style_ax(ax1, "Daily Attendance", "Date", "Count")
    ax1.tick_params(axis="x", rotation=25, labelsize=8)

    # ── Panel 2: Per-student % bars ───────────────────────────
    ax2    = fig.add_subplot(gs[0, 2])
    colors = [
        COL_GR if p >= ELIGIBILITY_THRESHOLD else COL_RD
        for p in pct_df["Pct"]
    ]
    ax2.barh(pct_df["Name"], pct_df["Pct"],
             color=colors, edgecolor=DARK_BORD, height=0.5, zorder=3)
    ax2.axvline(ELIGIBILITY_THRESHOLD, color=COL_YL, linestyle="--", linewidth=1.2)
    _style_ax(ax2, "Attendance %", "(%)", "")
    ax2.set_xlim(0, 110)

    # ── Panel 3: Eligibility pie ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(DARK_SURF)
    ax3.pie(
        [above, below],
        labels=[f"≥{ELIGIBILITY_THRESHOLD}% (eligible)",
                f"<{ELIGIBILITY_THRESHOLD}% (at risk)"],
        colors=[COL_GR, COL_RD],
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        textprops={"color": COL_TEXT, "fontsize": 8},
    )
    ax3.set_title("Eligibility", color=COL_TEXT, fontsize=11, fontweight="bold")

    # ── Panel 4: KPI metrics ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(DARK_SURF)
    ax4.axis("off")
    kpis = [
        ("Total Students",   str(len(all_names)),    COL_BL),
        ("Total Days",        str(total_days),         COL_MG),
        ("Avg Attendance",    f"{avg_pct:.1f}%",       COL_YL),
        (f"Eligible (≥{ELIGIBILITY_THRESHOLD}%)", str(above), COL_GR),
        (f"At Risk (<{ELIGIBILITY_THRESHOLD}%)",  str(below), COL_RD),
        ("Latest Session",    latest_day,              COL_DIM),
        ("Present (latest)",  str(latest_present),     COL_GR),
    ]
    for i, (label, value, color) in enumerate(kpis):
        y_pos = 0.92 - i * 0.135
        ax4.text(0.02, y_pos, label, transform=ax4.transAxes,
                 color=COL_DIM, fontsize=8, va="top")
        ax4.text(0.98, y_pos, value, transform=ax4.transAxes,
                 color=color, fontsize=10, fontweight="bold",
                 va="top", ha="right")
        if i < len(kpis) - 1:
            ax4.plot(
                [0.02, 0.98], [y_pos - 0.08, y_pos - 0.08],
                color=DARK_BORD, linewidth=0.5,
                transform=ax4.transAxes, clip_on=False,
            )
    ax4.set_title("Key Metrics", color=COL_TEXT, fontsize=11, fontweight="bold")

    # ── Panel 5: Student summary table ───────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(DARK_SURF)
    ax5.axis("off")
    tdata = [
        [
            row["Name"],
            str(int(row["Days"])),
            f"{row['Pct']}%",
            "✓" if row["Pct"] >= ELIGIBILITY_THRESHOLD else "✗",
        ]
        for _, row in pct_df.sort_values("Name").iterrows()
    ]
    tbl = ax5.table(
        cellText=tdata,
        colLabels=["Name", "Days", "%", "OK"],
        cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        bg = DARK_SURF if r % 2 == 0 else "#1c2128"
        cell.set_facecolor(bg)
        cell.set_edgecolor(DARK_BORD)
        if r == 0:
            cell.set_facecolor("#21262d")
            cell.get_text().set_color(COL_TEXT)
            cell.get_text().set_fontweight("bold")
        else:
            cell.get_text().set_color(COL_TEXT)
    ax5.set_title("Student Summary", color=COL_TEXT, fontsize=11, fontweight="bold")

    path = os.path.join(REPORT_DIR, "5_dashboard.png")
    _save(path)
    plt.show()
    ok("5_dashboard.png")


# ══════════════════════════════════════════════════════════════
#  Text report
# ══════════════════════════════════════════════════════════════

def generate_text_report(
    df: pd.DataFrame,
    pct_df: pd.DataFrame,
    all_names: list,
) -> None:
    """Write a structured plain-text attendance report to reports/."""
    total = df["Date"].nunique()
    path  = os.path.join(REPORT_DIR, "attendance_report.txt")

    def sep(char: str = "═", n: int = 60) -> str:
        return char * n

    with open(path, "w", encoding="utf-8") as f:
        # Header
        f.write(sep() + "\n")
        f.write("  SMART ATTENDANCE SYSTEM — OFFICIAL REPORT\n")
        f.write(sep() + "\n")
        f.write(f"  Generated  :  {datetime.now().strftime('%d %B %Y, %H:%M:%S')}\n")
        f.write(f"  Total Days :  {total}\n")
        f.write(f"  Students   :  {len(all_names)}\n")
        f.write(sep() + "\n\n")

        # Per-student summary
        f.write(
            f"  {'Name':<22} {'Present':>8} {'Absent':>8} "
            f"{'%':>7}   {'Status'}\n"
        )
        f.write("  " + sep("─", 56) + "\n")
        for _, row in pct_df.sort_values("Name").iterrows():
            absent = total - int(row["Days"])
            status = (
                "✔ ELIGIBLE" if row["Pct"] >= ELIGIBILITY_THRESHOLD
                else "✖ AT RISK"
            )
            f.write(
                f"  {row['Name']:<22} {int(row['Days']):>8} {absent:>8} "
                f"{row['Pct']:>6.1f}%   {status}\n"
            )

        # Day-wise log
        f.write("\n" + sep() + "\n  Day-wise Log\n" + sep() + "\n")
        for date in sorted(df["Date"].unique()):
            dstr   = pd.to_datetime(date).strftime("%A, %d %B %Y")
            day_df = df[df["Date"] == date].sort_values("Student Name")
            f.write(f"\n  {dstr}\n  " + sep("─", 40) + "\n")
            for _, row in day_df.iterrows():
                icon = "✔" if row["Status"] == "Present" else "✖"
                f.write(
                    f"    {icon}  {row['Student Name']:<22} "
                    f"{row['Status']:<10}  {row['Time']}\n"
                )

    ok("attendance_report.txt")


# ══════════════════════════════════════════════════════════════
#  Terminal attendance summary
# ══════════════════════════════════════════════════════════════

def print_terminal_report(
    df: pd.DataFrame,
    pct_df: pd.DataFrame,
    all_names: list,
) -> None:
    """Print a formatted attendance summary table to the terminal."""
    total = df["Date"].nunique()
    section("Attendance Summary Table")

    print(
        f"  {DM}  {'Name':<20} {'Present':>8} {'Absent':>8} "
        f"{'%':>8}   Status{R}"
    )
    print(
        f"  {DM}  {'─'*18} {'─'*8} {'─'*8} {'─'*8}   {'─'*10}{R}"
    )

    for _, row in pct_df.sort_values("Name").iterrows():
        absent = total - int(row["Days"])
        color  = GR if row["Pct"] >= ELIGIBILITY_THRESHOLD else RD
        status = (
            f"{GR}✔ ELIGIBLE{R}"
            if row["Pct"] >= ELIGIBILITY_THRESHOLD
            else f"{RD}✖ AT RISK{R}"
        )
        print(
            f"  {color}●{R}  {WH}{row['Name']:<20}{R} "
            f"{GR}{int(row['Days']):>8}{R} {RD}{absent:>8}{R} "
            f"{color}{row['Pct']:>7.1f}%{R}   {status}"
        )

    print(f"\n  {DM}Total days recorded: {total}{R}")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner()
    df, all_names = load_data()
    info(
        f"Loaded {len(df)} record(s)  |  "
        f"{df['Date'].nunique()} day(s)  |  "
        f"{len(all_names)} student(s)\n"
    )

    section("Generating Charts")
    plot_daily(df)
    pct_df = plot_student_percent(df, all_names)
    plot_pie(df, all_names)
    plot_heatmap(df, all_names)
    plot_dashboard(df, pct_df, all_names)

    section("Generating Text Report")
    generate_text_report(df, pct_df, all_names)
    print_terminal_report(df, pct_df, all_names)

    print(f"\n  {BG_GR} ALL DONE {R}  All reports saved in {CY}{REPORT_DIR}/{R}\n")
