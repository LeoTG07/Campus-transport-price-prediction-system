"""
generate_flowchart.py
======================
Generates a standard System Flowchart (terminals, processes, decisions)
for the Campus Transport Price Prediction System, suitable for Chapter 4.5
"System Flowchart" of the project report.

This is deliberately a separate script from train_model.py because a
System Flowchart (classic ANSI flowchart symbols) is a different artefact
from the UML Activity Diagram already produced for Chapter 3.

Run with:
    python generate_flowchart.py

Output:
    outputs/system_flowchart.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.lines import Line2D

OUTPUT_PATH = "outputs/system_flowchart.png"

NAVY = "#0B2545"
AMBER = "#F5A623"
GREEN = "#2F9E44"
LIGHT = "#EAF0FA"
WHITE = "#FFFFFF"

fig, ax = plt.subplots(figsize=(8.5, 13))
ax.set_xlim(0, 10)
ax.set_ylim(0, 34)
ax.axis("off")


def terminal(cx, cy, text, w=4.4, h=1.2, color=NAVY):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        linewidth=1.5, edgecolor=color, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", color="white", fontsize=10, fontweight="bold")


def process(cx, cy, text, w=4.8, h=1.3, color=LIGHT, edge=NAVY):
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="square,pad=0.02",
        linewidth=1.5, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", color=NAVY, fontsize=9.5)


def decision(cx, cy, text, w=5.2, h=1.9, color="#FFF4E0", edge=AMBER):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    poly = Polygon(pts, closed=True, linewidth=1.5, edgecolor=edge, facecolor=color)
    ax.add_patch(poly)
    ax.text(cx, cy, text, ha="center", va="center", color=NAVY, fontsize=9, fontweight="bold")


def arrow(x1, y1, x2, y2, text=None, text_dx=0.35):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6, shrinkA=0, shrinkB=0),
    )
    if text:
        ax.text((x1 + x2) / 2 + text_dx, (y1 + y2) / 2, text, fontsize=8.5, color="#7A1F00", fontweight="bold")


# ---- Flow ----
terminal(5, 33, "START")
arrow(5, 32.4, 5, 31.4)

process(5, 30.6, "Student opens the web application\n(Streamlit interface)")
arrow(5, 29.9, 5, 28.9)

process(5, 28.1, "Student enters Current Fuel Price (NGN/L)\nand selects Festive Period (Yes/No)")
arrow(5, 27.4, 5, 26.4)

process(5, 25.6, "System receives input\n(Predict Fare button clicked)")
arrow(5, 24.9, 5, 23.9)

decision(5, 22.6, "Is input\nvalid?")
arrow(5, 21.6, 5, 20.6)

process(5, 19.8, "System preprocesses input\n(type-check, scale fuel price)")
arrow(5, 19.1, 5, 18.1)

process(5, 17.3, "System loads the trained\nprediction model (best_model.pkl)")
arrow(5, 16.6, 5, 15.6)

process(5, 14.8, "Model generates predicted\ncampus transport fare")
arrow(5, 14.1, 5, 13.1)

process(5, 12.3, "System displays the predicted fare\non the results card")
arrow(5, 11.6, 5, 10.6)

decision(5, 9.3, "Predict\nanother fare?")
arrow(5, 8.3, 5, 7.3)

terminal(5, 6.5, "END")

def elbow_path(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=NAVY, lw=1.4)
    ax.annotate(
        "", xy=points[-1], xytext=points[-2],
        arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.4, shrinkA=0, shrinkB=0),
    )


# NO branch from validity check -> error message -> back to input
arrow(7.6, 22.6, 8.7, 22.6)
process(8.7, 22.6, "Display error\nmessage", w=2.6, h=1.5)
ax.text(7.75, 22.85, "No", fontsize=8.5, color="#7A1F00", fontweight="bold")
elbow_path([(8.7, 21.85), (9.6, 21.85), (9.6, 28.1), (7.4, 28.1)])
ax.text(9.65, 25, "Back to input", fontsize=8, color=NAVY, style="italic", rotation=90, va="center")
ax.text(5.6, 21.75, "Yes", fontsize=8.5, color="#7A1F00", fontweight="bold")

# YES branch from "predict another?" loops back to input entry
elbow_path([(2.4, 9.3), (0.6, 9.3), (0.6, 28.1), (2.6, 28.1)])
ax.text(0.35, 18, "Yes \u2013 predict again", fontsize=8, color=NAVY, style="italic", rotation=90, va="center")
ax.text(5.6, 8.5, "No", fontsize=8.5, color="#7A1F00", fontweight="bold")

ax.set_title(
    "Fig. 4.x — System Flowchart:\nCampus Transport Price Prediction System",
    fontsize=13, fontweight="bold", color=NAVY, pad=14,
)

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
print(f"Flowchart saved to {OUTPUT_PATH}")
