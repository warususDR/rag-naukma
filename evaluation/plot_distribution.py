import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.family"] = "DejaVu Sans"

EVAL_DIR = Path(__file__).parent

labels = ["Запитання про\nконкретні факти", "Запитання на\nміркування", "Граничні\nвипадки"]
values = [90, 35, 5]
colors = ["#4e79a7", "#59a14f", "#f28e2b"]
total = sum(values)

fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Розподіл типів питань у датасеті",
    fontsize=14, fontweight="bold"
)

bars = ax_bar.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

for bar, val in zip(bars, values):
    ax_bar.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        str(val),
        ha="center", va="bottom", fontsize=13, fontweight="bold"
    )

ax_bar.set_title("Кількість запитань за категорією", fontsize=12, pad=10)
ax_bar.set_ylabel("Кількість запитань", fontsize=11)
ax_bar.set_ylim(0, max(values) + 12)
ax_bar.tick_params(axis="x", labelsize=11)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_bar.set_axisbelow(True)

explode = (0.03, 0.03, 0.07)
wedges, texts, autotexts = ax_pie.pie(
    values,
    labels=labels,
    colors=colors,
    explode=explode,
    autopct=lambda p: f"{p:.1f}%\n({int(round(p * total / 100))})",
    startangle=140,
    pctdistance=0.65,
    textprops={"fontsize": 10},
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")

ax_pie.set_title("Частка за категорією", fontsize=12, pad=10)

plt.tight_layout()
output_path = EVAL_DIR / "dataset_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
