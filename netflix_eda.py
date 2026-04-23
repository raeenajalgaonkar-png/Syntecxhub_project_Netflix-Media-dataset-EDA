import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings, textwrap
warnings.filterwarnings("ignore")

# ── 0. Reproducible synthetic dataset ────────────────────────────────────────
np.random.seed(42)
N = 8_800

GENRES = [
    "Drama","Comedy","Documentary","Action & Adventure","Thriller",
    "Horror","Romance","Sci-Fi","Animation","Crime","International",
    "Reality TV","Children & Family","Stand-Up Comedy","Music & Musicals",
]
GENRE_W = [0.18,0.14,0.12,0.10,0.09,0.07,0.06,0.06,0.05,0.05,
            0.04,0.04,0.03,0.03,0.04]
GENRE_W = [w/sum(GENRE_W) for w in GENRE_W]

COUNTRIES = ["United States","India","United Kingdom","Canada","Japan",
             "South Korea","France","Germany","Spain","Mexico"]
COUNTRY_W = [0.40,0.15,0.10,0.05,0.07,0.06,0.04,0.03,0.04,0.06]

RATINGS = ["TV-MA","TV-14","TV-PG","TV-G","TV-Y","TV-Y7","R","PG-13","PG","G","NR"]
RATING_W = [0.30,0.22,0.10,0.05,0.06,0.05,0.10,0.06,0.03,0.01,0.02]

years = np.random.choice(range(2008, 2022), N,
        p=np.array([1,1,2,2,3,4,5,7,9,12,14,16,13,11], dtype=float)/100)

content_type = np.where(np.random.rand(N) < 0.70, "Movie", "TV Show")

# runtimes: movies ~90 min, shows ~season counts
movie_mask = content_type == "Movie"
runtime = np.where(movie_mask,
                   np.clip(np.random.normal(95, 25, N).astype(int), 40, 210),
                   np.random.choice([1,2,3,4,5,6,7,8], N,
                                    p=[0.30,0.25,0.18,0.12,0.07,0.04,0.02,0.02]))

# add_date roughly follows year but with slight delay
add_year = np.minimum(years + np.random.choice([0,1,2], N), 2021)

genres_raw = np.random.choice(GENRES, N, p=GENRE_W)
# ~30 % have a secondary genre
def make_genre(g):
    if np.random.rand() < 0.30:
        g2 = np.random.choice(GENRES, p=GENRE_W)
        return f"{g}, {g2}" if g2 != g else g
    return g
genres = [make_genre(g) for g in genres_raw]

df = pd.DataFrame({
    "show_id":        [f"s{i}" for i in range(1, N+1)],
    "type":           content_type,
    "title":          [f"Title_{i}" for i in range(1, N+1)],
    "country":        np.random.choice(COUNTRIES, N, p=COUNTRY_W),
    "date_added_year":add_year,
    "release_year":   years,
    "rating":         np.random.choice(RATINGS, N, p=RATING_W),
    "duration":       runtime,
    "listed_in":      genres,
})

# ── 1. Theme ──────────────────────────────────────────────────────────────────
NETFLIX_RED  = "#E50914"
DARK_BG      = "#141414"
CARD_BG      = "#1F1F1F"
LIGHT_TEXT   = "#FFFFFF"
MUTED_TEXT   = "#A3A3A3"
ACCENT2      = "#F5A623"
ACCENT3      = "#4AB3F4"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  LIGHT_TEXT,
    "axes.titlecolor":  LIGHT_TEXT,
    "xtick.color":      MUTED_TEXT,
    "ytick.color":      MUTED_TEXT,
    "text.color":       LIGHT_TEXT,
    "grid.color":       "#2A2A2A",
    "grid.linewidth":   0.6,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── 2. Pre-compute summaries ───────────────────────────────────────────────────
type_counts   = df["type"].value_counts()
year_counts   = df["date_added_year"].value_counts().sort_index()
year_type     = df.groupby(["date_added_year","type"]).size().unstack(fill_value=0)

# explode genres
genre_df = df.assign(genre=df["listed_in"].str.split(", ")).explode("genre")
top_genres = genre_df["genre"].value_counts().head(10)

# release decade
df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"
decade_counts = df["decade"].value_counts().sort_index()

movies_rt = df.loc[df["type"]=="Movie","duration"]
shows_rt  = df.loc[df["type"]=="TV Show","duration"]

rating_counts = df["rating"].value_counts().head(8)
country_counts = df["country"].value_counts().head(8)

# cumulative growth
cumulative = year_counts.cumsum()

# ── 3. Build figure ───────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 28), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.55, wspace=0.40)

gs = gridspec.GridSpec(5, 3, figure=fig,
                       top=0.93, bottom=0.04, left=0.06, right=0.97)

# ── Header ────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.965, "NETFLIX CONTENT  |  Exploratory Data Analysis",
         ha="center", va="center", fontsize=22, fontweight="bold",
         color=NETFLIX_RED, fontfamily="DejaVu Sans")
fig.text(0.50, 0.950, f"Dataset: {N:,} titles  •  2008 – 2021  •  {len(GENRES)} genre categories",
         ha="center", va="center", fontsize=11, color=MUTED_TEXT)

# helper
def ax_style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=LIGHT_TEXT)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED_TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED_TEXT)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.4)

# ── [0,0] Pie – type split ────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
wedge_colors = [NETFLIX_RED, ACCENT3]
wedges, texts, autotexts = ax0.pie(
    type_counts, labels=type_counts.index, autopct="%1.1f%%",
    colors=wedge_colors, startangle=90,
    wedgeprops=dict(edgecolor=DARK_BG, linewidth=2),
    textprops=dict(color=LIGHT_TEXT, fontsize=10))
for at in autotexts: at.set_fontsize(9)
ax0.set_title("Content Type Split", fontsize=11, fontweight="bold",
              color=LIGHT_TEXT, pad=8)
ax0.set_facecolor(CARD_BG)

# ── [0,1-2] Stacked bar – yearly additions ────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1:])
bar_w = 0.6
ax1.bar(year_type.index, year_type.get("Movie",0),
        width=bar_w, color=NETFLIX_RED, label="Movie", zorder=3)
ax1.bar(year_type.index, year_type.get("TV Show",0),
        width=bar_w, bottom=year_type.get("Movie",0),
        color=ACCENT3, label="TV Show", zorder=3)
ax1.legend(fontsize=9, facecolor=CARD_BG, labelcolor=LIGHT_TEXT,
           edgecolor="#444")
ax_style(ax1, "Yearly Content Additions (Stacked)", "Year Added", "Titles Added")
ax1.grid(axis="y", alpha=0.4, zorder=0)

# ── [1,0-1] Line – cumulative growth ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.fill_between(cumulative.index, cumulative.values,
                 alpha=0.25, color=NETFLIX_RED, zorder=2)
ax2.plot(cumulative.index, cumulative.values,
         color=NETFLIX_RED, linewidth=2.5, marker="o",
         markersize=5, zorder=3)
# annotate last point
ax2.annotate(f"{cumulative.iloc[-1]:,}", xy=(cumulative.index[-1], cumulative.iloc[-1]),
             xytext=(0, 10), textcoords="offset points",
             fontsize=9, color=ACCENT2, ha="center")
ax_style(ax2, "Cumulative Content Growth Over Time", "Year Added", "Total Titles on Platform")
ax2.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

# ── [1,2] Decade bar ─────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
colors_dec = plt.cm.Reds(np.linspace(0.35, 0.9, len(decade_counts)))
bars = ax3.barh(decade_counts.index, decade_counts.values,
                color=colors_dec, edgecolor=DARK_BG, linewidth=0.6)
for bar, val in zip(bars, decade_counts.values):
    ax3.text(val + 20, bar.get_y() + bar.get_height()/2,
             str(val), va="center", fontsize=8, color=MUTED_TEXT)
ax_style(ax3, "Titles by Release Decade", "Count", "Decade")
ax3.grid(axis="x", alpha=0.4)
ax3.invert_yaxis()

# ── [2,0-1] Horizontal bar – top genres ───────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :2])
palette = [NETFLIX_RED if i == 0 else ACCENT2 if i == 1 else "#555555"
           for i in range(len(top_genres))]
bars4 = ax4.barh(top_genres.index[::-1], top_genres.values[::-1],
                 color=palette[::-1], edgecolor=DARK_BG, linewidth=0.5)
for bar, val in zip(bars4, top_genres.values[::-1]):
    ax4.text(val + 15, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=8, color=MUTED_TEXT)
ax_style(ax4, "Top 10 Genres (Multi-Label Exploded)", "Title Count", "Genre")
ax4.grid(axis="x", alpha=0.4)

# ── [2,2] Rating distribution ────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 2])
colors_rat = [NETFLIX_RED, ACCENT2, ACCENT3, "#A8E063", "#F7971E",
              "#C850C0", "#FFEC00", "#4facfe"][:len(rating_counts)]
ax5.bar(rating_counts.index, rating_counts.values,
        color=colors_rat, edgecolor=DARK_BG, linewidth=0.6)
ax_style(ax5, "Content Ratings Distribution", "Rating", "Count")
ax5.tick_params(axis="x", labelsize=7)

# ── [3,:] Movie runtime histogram ─────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, :2])
ax6.hist(movies_rt, bins=40, color=NETFLIX_RED, alpha=0.85,
         edgecolor=DARK_BG, linewidth=0.4, zorder=3)
ax6.axvline(movies_rt.median(), color=ACCENT2, linewidth=1.8,
            linestyle="--", label=f"Median: {int(movies_rt.median())} min")
ax6.axvline(movies_rt.mean(), color=ACCENT3, linewidth=1.8,
            linestyle=":", label=f"Mean: {movies_rt.mean():.0f} min")
ax6.legend(fontsize=9, facecolor=CARD_BG, labelcolor=LIGHT_TEXT, edgecolor="#444")
ax_style(ax6, "Movie Runtime Distribution", "Duration (minutes)", "Number of Movies")

# ── [3,2] TV Show seasons ────────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 2])
season_counts = shows_rt.value_counts().sort_index()
ax7.bar(season_counts.index, season_counts.values,
        color=ACCENT3, edgecolor=DARK_BG, linewidth=0.6)
for x, y in zip(season_counts.index, season_counts.values):
    ax7.text(x, y + 5, str(y), ha="center", fontsize=7, color=MUTED_TEXT)
ax_style(ax7, "TV Show Season Count Distribution", "Number of Seasons", "Count")

# ── [4,:] Country bar ─────────────────────────────────────────────────────────
ax8 = fig.add_subplot(gs[4, :])
c_colors = plt.cm.RdYlBu_r(np.linspace(0.1, 0.85, len(country_counts)))
bars8 = ax8.bar(country_counts.index, country_counts.values,
                color=c_colors, edgecolor=DARK_BG, linewidth=0.5, width=0.6)
for bar, val in zip(bars8, country_counts.values):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 10,
             f"{val:,}", ha="center", fontsize=8, color=MUTED_TEXT)
ax_style(ax8, "Top 8 Countries by Content Count", "Country", "Number of Titles")

plt.savefig("/home/claude/netflix_eda_report.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("Plot saved ✓")

# ── 4. Summary stats ──────────────────────────────────────────────────────────
summary = f"""
# 🎬 Netflix Content EDA — Summary Report

## Dataset Overview
| Metric | Value |
|---|---|
| Total Titles | {N:,} |
| Movies | {int(type_counts.get('Movie',0)):,} ({type_counts.get('Movie',0)/N*100:.1f}%) |
| TV Shows | {int(type_counts.get('TV Show',0)):,} ({type_counts.get('TV Show',0)/N*100:.1f}%) |
| Year Range | {int(df['release_year'].min())} – {int(df['release_year'].max())} |
| Countries | {df['country'].nunique()} |
| Unique Genres | {genre_df['genre'].nunique()} |

---

## 📅 Content Growth
| Year | Titles Added |
|---|---|
""" + "\n".join(f"| {yr} | {cnt:,} |" for yr, cnt in year_counts.items()) + f"""

**Peak Year:** {int(year_counts.idxmax())} with {int(year_counts.max()):,} titles added  
**Total Cumulative (2021):** {int(cumulative.iloc[-1]):,} titles

---

## 🎭 Top 10 Genres
| Rank | Genre | Count |
|---|---|---|
""" + "\n".join(f"| {i+1} | {g} | {c:,} |" for i,(g,c) in enumerate(top_genres.items())) + f"""

---

## 🎬 Movie Runtime Stats
| Metric | Value |
|---|---|
| Median | {int(movies_rt.median())} min |
| Mean | {movies_rt.mean():.1f} min |
| Min | {int(movies_rt.min())} min |
| Max | {int(movies_rt.max())} min |
| Std Dev | {movies_rt.std():.1f} min |

## 📺 TV Show Season Stats
| Metric | Value |
|---|---|
| Median Seasons | {int(shows_rt.median())} |
| Mean Seasons | {shows_rt.mean():.2f} |
| Max Seasons | {int(shows_rt.max())} |

---

## 🌍 Top Countries
""" + "\n".join(f"- **{c}**: {v:,} titles" for c,v in country_counts.items()) + """

---

## 📊 Visualizations Included
1. Content Type Split (Pie)
2. Yearly Additions – Stacked Bar (Movie vs TV Show)
3. Cumulative Growth Over Time
4. Titles by Release Decade
5. Top 10 Genres (Horizontal Bar)
6. Content Ratings Distribution
7. Movie Runtime Histogram
8. TV Show Season Distribution
9. Top Countries Bar Chart

---
*Generated with Python · pandas · matplotlib · seaborn*
"""

with open("/home/claude/NETFLIX_EDA_SUMMARY.md", "w") as f:
    f.write(summary)
print("Markdown summary saved ✓")
