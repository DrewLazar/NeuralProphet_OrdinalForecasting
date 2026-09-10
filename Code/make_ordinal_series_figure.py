"""
Figure: two-week windows of the two ordinal series.

Plots ONLY the ordinal labels, which is what the model observes; the underlying
PM2.5 and kW measurements are not used. y-axis tick labels give the concentration
or load band defining each category, which also displays the contrast in
cut-point spacing: Beijing's gaps are 40 and 75 ug/m3 (unequal), the FSB bands
are 45 kW apart (equal, being mean +/- 0.7 SD).

Writes figures/ordinal_series.pdf (and .png for inspection).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BEIJING_CSV = os.environ.get("BEIJING_CSV", "BeijingAQI_data.csv")
ENERGY_CSV  = os.environ.get("ENERGY_CSV",  "energy_data_excerpt.csv")
OUTDIR      = os.environ.get("OUTDIR", "figures")
COLOR       = "#0072B2"          # matches the other decomposition figures
os.makedirs(OUTDIR, exist_ok=True)

# Two-week windows: Beijing mid-record, FSB during the autumn teaching term.
BEIJING_WINDOW = ("2014-03-03", "2014-03-17")
ENERGY_WINDOW  = ("2024-10-07", "2024-10-21")


def load(csv, window):
    d = pd.read_csv(csv)
    d["datetime"] = pd.to_datetime(d["datetime"])
    lo, hi = window
    return d[(d.datetime >= lo) & (d.datetime < hi)]


def hour_of_day_spread(csv, col):
    """Range of the hour-of-day mean category: the cyclicality statistic in the caption."""
    d = pd.read_csv(csv)
    d["datetime"] = pd.to_datetime(d["datetime"])
    h = d.groupby(d.datetime.dt.hour)[col].mean()
    return h.max() - h.min()


bw = load(BEIJING_CSV, BEIJING_WINDOW)
ew = load(ENERGY_CSV,  ENERGY_WINDOW)
b_spread = hour_of_day_spread(BEIJING_CSV, "aqi_category")
e_spread = hour_of_day_spread(ENERGY_CSV,  "load_category")
print(f"beijing window n={len(bw)}  hour-of-day spread {b_spread:.2f}")
print(f"fsb window     n={len(ew)}  hour-of-day spread {e_spread:.2f}")

panels = [
    # Beijing names follow HJ 633-2012; categories 3 and 4 each merge two of its
    # six levels (75-115 lightly + 115-150 moderately; 150+ heavily and above).
    (bw, "aqi_category",
     rf"(a) Beijing AQI, 3--16 March 2014   (hour-of-day range $={b_spread:.2f}$)",
     ["Excellent\n($\\leq$35)", "Good\n(35--75)",
      "Light--mod.\n(75--150)", "Heavy\n($>$150)"],
     r"PM$_{2.5}$ category"),
    (ew, "load_category",
     rf"(b) FSB campus load, 7--20 October 2024   (hour-of-day range $={e_spread:.2f}$)",
     ["Low\n($\\leq$370)", "Below avg.\n(370--415)",
      "Above avg.\n(415--460)", "High\n($>$460)"],
     "Load category (kW)"),
]

# Authored at the final printed width (~\textwidth) so fonts are not shrunk on placement.
fig, axes = plt.subplots(2, 1, figsize=(6.9, 3.9))
for ax, (d, col, title, ylabels, ylab) in zip(axes, panels):
    for day, _ in d.groupby(d.datetime.dt.normalize()):     # shade weekends
        if day.dayofweek >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1), color="0.90", lw=0, zorder=0)
    ax.step(d.datetime, d[col], where="post", color=COLOR, lw=0.9, zorder=3)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(ylabels, fontsize=6.2, linespacing=0.95)
    ax.set_ylim(0.6, 4.5)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="y", color="0.92", lw=0.6, zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

axes[1].set_xlabel("Date", fontsize=8.5)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUTDIR, f"ordinal_series.{ext}"),
                dpi=145, bbox_inches="tight")
print(f"wrote {OUTDIR}/ordinal_series.pdf")