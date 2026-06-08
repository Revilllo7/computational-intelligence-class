"""
task04/titanic_apriori.py
==========================
Analiza reguł asocjacyjnych na danych pasażerów Titanica.

Pytanie badawcze:
  Czy płeć, klasa podróży i wiek mają związek z przeżywalnością?

Dane wejściowe: task04/titanic_raw.csv (pobierany automatycznie)
Dane robocze:   task04/titanic.csv  (tylko 4 kolumny: Class, Sex, Age, Survived)

Metodologia:
  1. Wczytanie danych i redukcja do 4 kolumn: Class, Sex, Age, Survived
  2. Dyskretyzacja zmiennych ciągłych (Age → child/adult)
  3. One-hot encoding (mlxtend TransactionEncoder)
  4. Algorytm Apriori (min_support=0.005)
  5. Reguły asocjacyjne (min_confidence=0.8), posortowane wg. ufności
  6. Filtracja najciekawszych reguł — wskazujących na przeżywalność
  7. Wizualizacje: scatter (support vs confidence), heatmap, bar chart

Wymagania: pandas, mlxtend, matplotlib, seaborn
"""

import warnings
from pathlib import Path
from typing import Any, cast

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")

matplotlib.use("Agg")

# ─────────────────────────────────────────────────────────────────────────────
#  KOLORY I STYL
# ─────────────────────────────────────────────────────────────────────────────
BG_DARK = "#1a1a2e"
BG_PANEL = "#16213e"
ACCENT1 = "#4e9af1"
ACCENT2 = "#f0a500"
ACCENT3 = "#3ecf8e"
ACCENT4 = "#f45f74"
TEXT_COL = "#c9d1d9"
GRID_COL = "#21262d"

plt.rcParams.update(
    {
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": TEXT_COL,
        "xtick.color": TEXT_COL,
        "ytick.color": TEXT_COL,
        "text.color": TEXT_COL,
        "grid.color": GRID_COL,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "legend.facecolor": "#0f3460",
        "legend.edgecolor": ACCENT1,
    }
)

# ─────────────────────────────────────────────────────────────────────────────
#  1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path.resolve(Path(__file__).parent)
TASK_DIR = Path.resolve(Path(__file__)).parent.parent
RAW_CSV = TASK_DIR / "data" / "titanic_raw.csv"
OUT_CSV = TASK_DIR / "output" / "titanic.csv"
# OUT_CSV.mkdir(exist_ok=True)
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def prepare_data() -> pd.DataFrame:
    """Wczytuje, redukuje do 4 kolumn i dyskretyzuje dane."""
    df_raw = pd.read_csv(RAW_CSV)

    # Mapowanie kolumn na wymagane 4: Class, Sex, Age, Survived
    df = pd.DataFrame()
    df["Class"] = df_raw["Pclass"].replace(
        {
            1: "1st",
            2: "2nd",
            3: "3rd",
        }
    )
    df["Sex"] = df_raw["Sex"]  # male / female
    # Dyskretyzacja wieku: child (≤12), adult (>12), usuń NaN
    df["Age"] = df_raw["Age"].apply(
        lambda a: "child" if pd.notna(a) and a <= 12 else "adult" if pd.notna(a) else None
    )
    df["Survived"] = df_raw["Survived"].replace(
        {
            0: "not_survived",
            1: "survived",
        }
    )

    # Usuwamy wiersze z brakującymi wartościami
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Zapisz wynikowy plik titanic.csv (tylko 4 kolumny)
    df.to_csv(OUT_CSV, index=False)
    print(f"[data] titanic.csv zapisany: {df.shape[0]} wierszy, kolumny: {list(df.columns)}")
    print(df.head(5).to_string())
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2. ONE-HOT ENCODING → APRIORI
# ─────────────────────────────────────────────────────────────────────────────


def run_apriori(df: pd.DataFrame):
    """
    Tworzy transakcje z każdego wiersza jako zbioru wartości atrybutów,
    one-hot koduje je i uruchamia algorytm Apriori.
    """
    # Każda transakcja = lista wartości atrybutów w postaci "Kolumna=Wartość"
    transactions = []
    for class_, sex, age, survived in df.itertuples(index=False, name=None):
        transactions.append(
            [
                f"Class={class_}",
                f"Sex={sex}",
                f"Age={age}",
                f"Survived={survived}",
            ]
        )

    # TransactionEncoder → binarna macierz
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    if isinstance(te_array, csr_matrix):
        te_array = te_array.toarray()
    df_encoded = pd.DataFrame(cast(Any, te_array), columns=te.columns_)

    print(f"[apriori] Zbiór {df_encoded.shape[0]} transakcji, {df_encoded.shape[1]} itemów:")
    print(f"  Itemy: {list(df_encoded.columns)}\n")

    # Algorytm Apriori: min_support = 0.005 (0.5%)
    frequent_itemsets = apriori(
        df_encoded,
        min_support=0.005,
        use_colnames=True,
    )
    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    print(f"[apriori] Znalezione częste itemsety: {len(frequent_itemsets)}")
    print(frequent_itemsets.sort_values("support", ascending=False).head(10).to_string())
    print()

    return frequent_itemsets, df_encoded


# ─────────────────────────────────────────────────────────────────────────────
#  3. REGUŁY ASOCJACYJNE
# ─────────────────────────────────────────────────────────────────────────────


def extract_rules(frequent_itemsets):
    """
    Generuje reguły asocjacyjne z min_confidence=0.8 i sortuje wg ufności.
    """
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=0.8,
        num_itemsets=len(frequent_itemsets),
    )
    rules = rules.sort_values("confidence", ascending=False).reset_index(drop=True)

    print(f"[rules] Liczba reguł (confidence ≥ 0.8): {len(rules)}")
    print("\n--- TOP 15 reguł (posortowane wg ufności) ---")
    for counter, (_i, row) in enumerate(rules.head(15).iterrows()):
        ant = ", ".join(sorted(row["antecedents"]))
        con = ", ".join(sorted(row["consequents"]))
        print(
            f"  {counter + 1:2d}. [{ant}] → [{con}]"
            f"  sup={row['support']:.3f}  conf={row['confidence']:.3f}  lift={row['lift']:.2f}"
        )
    print()
    return rules


# ─────────────────────────────────────────────────────────────────────────────
#  4. FILTRACJA CIEKAWYCH REGUŁ (związanych z przeżywalnością)
# ─────────────────────────────────────────────────────────────────────────────


def filter_survival_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Filtruje reguły, których konsekwentem jest Survived=survived lub Survived=not_survived.
    """

    def consequent_is_survived(cons):
        return any(c.startswith("Survived=") for c in cons)

    survival_rules = rules[rules["consequents"].apply(consequent_is_survived)].copy()
    survival_rules = survival_rules.sort_values("confidence", ascending=False).reset_index(  # type: ignore[reportCallIssue]
        drop=True
    )

    print(f"[filter] Reguły dotyczące przeżywalności: {len(survival_rules)}")
    print("\n--- Reguły → przeżywalność ---")
    for counter, (_i, row) in enumerate(survival_rules.iterrows()):
        ant = ", ".join(sorted(row["antecedents"]))
        con = ", ".join(sorted(row["consequents"]))
        print(
            f"  {counter + 1:2d}. [{ant}] → [{con}]"
            f"  sup={row['support']:.3f}  conf={row['confidence']:.3f}  lift={row['lift']:.2f}"
        )
    print()
    return survival_rules


# ─────────────────────────────────────────────────────────────────────────────
#  5. WIZUALIZACJE
# ─────────────────────────────────────────────────────────────────────────────


def plot_support_confidence(rules: pd.DataFrame, survival_rules: pd.DataFrame) -> None:
    """Scatter: support vs confidence, kolorowane wg lift."""
    _fig, ax = plt.subplots(figsize=(10, 6))

    # Wszystkie reguły — szare tło
    sc_all = ax.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="coolwarm",
        alpha=0.5,
        s=40,
        zorder=2,
    )

    # Reguły o przeżywalności — podświetlone
    if not survival_rules.empty:
        for _, row in survival_rules.iterrows():
            label = ", ".join(sorted(row["consequents"]))
            color = ACCENT3 if "not_survived" not in label else ACCENT4
            ax.scatter(
                row["support"],
                row["confidence"],
                s=120,
                color=color,
                zorder=5,
                edgecolors="white",
                linewidth=0.8,
            )

    cbar = plt.colorbar(sc_all, ax=ax)
    cbar.set_label("Lift", color=TEXT_COL)
    cbar.ax.yaxis.set_tick_params(color=TEXT_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COL)

    survived_patch = mpatches.Patch(color=ACCENT3, label="→ survived")
    notsurv_patch = mpatches.Patch(color=ACCENT4, label="→ not_survived")
    all_patch = mpatches.Patch(color="#888888", alpha=0.6, label="Pozostałe reguły")
    ax.legend(handles=[survived_patch, notsurv_patch, all_patch], fontsize=9)

    ax.set_xlabel("Support (wsparcie)", fontsize=12)
    ax.set_ylabel("Confidence (ufność)", fontsize=12)
    ax.set_title(
        "Reguły asocjacyjne: Support vs Confidence\n(podświetlone reguły dot. przeżywalności)",
        fontsize=13,
        fontweight="bold",
        color="#e6edf3",
    )
    ax.grid(True)
    plt.tight_layout()
    path = PLOTS_DIR / "scatter_support_confidence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Zapisano → {path}")


def plot_survival_bar(df: pd.DataFrame) -> None:
    """Słupkowe: przeżywalność wg klasy i płci."""
    _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Przeżywalność wg klasy
    surv_class = df.groupby(["Class", "Survived"]).size().unstack(fill_value=0)
    surv_class.plot(
        kind="bar",
        ax=axes[0],
        color=[ACCENT4, ACCENT3],
        edgecolor="#000",
        width=0.6,
    )
    axes[0].set_title("Przeżywalność wg klasy podróży", fontweight="bold", color="#e6edf3")
    axes[0].set_xlabel("Klasa")
    axes[0].set_ylabel("Liczba pasażerów")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(["Nie przeżył", "Przeżył"], facecolor="#0f3460", edgecolor=ACCENT1)

    # Przeżywalność wg płci
    surv_sex = df.groupby(["Sex", "Survived"]).size().unstack(fill_value=0)
    surv_sex.plot(
        kind="bar",
        ax=axes[1],
        color=[ACCENT4, ACCENT3],
        edgecolor="#000",
        width=0.5,
    )
    axes[1].set_title("Przeżywalność wg płci", fontweight="bold", color="#e6edf3")
    axes[1].set_xlabel("Płeć")
    axes[1].set_ylabel("Liczba pasażerów")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(["Nie przeżył", "Przeżył"], facecolor="#0f3460", edgecolor=ACCENT1)

    for ax in axes:
        ax.set_facecolor(BG_PANEL)
        ax.grid(True, axis="y")

    plt.tight_layout()
    path = PLOTS_DIR / "bar_survival.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Zapisano → {path}")


def plot_heatmap_confidence(survival_rules: pd.DataFrame) -> None:
    """Heatmapa ufności reguł dot. przeżywalności dla top reguł."""
    if survival_rules.empty:
        return

    top = survival_rules.head(20).copy()
    top["antecedent_str"] = top["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    top["consequent_str"] = top["consequents"].apply(lambda x: ", ".join(sorted(x)))
    top["rule_label"] = top["antecedent_str"] + " → " + top["consequent_str"]

    _fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.45)))

    colors_conf = [ACCENT3 if "not_survived" not in r else ACCENT4 for r in top["consequent_str"]]
    bars = ax.barh(
        top["rule_label"], top["confidence"], color=colors_conf, edgecolor="#000", height=0.65
    )

    # Annotate lift
    for bar, (_, row) in zip(bars, top.iterrows(), strict=False):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"lift={row['lift']:.2f}  sup={row['support']:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=TEXT_COL,
        )

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Ufność (Confidence)", fontsize=11)
    ax.set_title(
        "Top reguły dotyczące przeżywalności (Confidence ≥ 0.8)",
        fontweight="bold",
        color="#e6edf3",
        fontsize=12,
    )
    ax.axvline(0.8, color=ACCENT2, linestyle="--", linewidth=1.2, label="min_confidence=0.8")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, axis="x")

    survived_patch = mpatches.Patch(color=ACCENT3, label="→ survived")
    notsurv_patch = mpatches.Patch(color=ACCENT4, label="→ not_survived")
    ax.legend(handles=[survived_patch, notsurv_patch], fontsize=9, loc="lower right")

    plt.tight_layout()
    path = PLOTS_DIR / "heatmap_survival_rules.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Zapisano → {path}")


def plot_lift_heatmap(survival_rules: pd.DataFrame) -> None:
    """Heatmapa lift dla kombinacji antecedent / consequent (Sex+Class → Survived)."""
    # Grupujemy reguły wg płci i klasy jako antecedentu
    combos = []
    for _, row in survival_rules.iterrows():
        ant = sorted(row["antecedents"])
        con = sorted(row["consequents"])
        combos.append(
            {
                "antecedent": " & ".join(ant),
                "consequent": " & ".join(con),
                "lift": row["lift"],
                "confidence": row["confidence"],
            }
        )

    if not combos:
        return

    df_combo = pd.DataFrame(combos).drop_duplicates(subset=["antecedent", "consequent"])
    pivot = df_combo.pivot_table(
        index="antecedent", columns="consequent", values="confidence", aggfunc="max"
    )

    if pivot.empty or pivot.shape[0] < 2:
        return

    _fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 2.5), max(4, pivot.shape[0] * 0.55)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
        linecolor="#30363d",
        cbar_kws={"label": "Confidence"},
    )
    ax.set_title(
        "Heatmapa ufności: Antecedens → Następnik\n(reguły dotyczące przeżywalności)",
        fontweight="bold",
        color="#e6edf3",
        fontsize=11,
    )
    ax.set_xlabel("Następnik (Consequent)", fontsize=10)
    ax.set_ylabel("Antecedens (Antecedent)", fontsize=10)
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    path = PLOTS_DIR / "lift_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Zapisano → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("ANALIZA REGUŁ ASOCJACYJNYCH — TITANIC (Apriori)")
    print("=" * 70)
    print()

    # 1. Dane
    df = prepare_data()

    # Statystyki opisowe
    total = len(df)
    print(f"[stats] Pasażerów po czyszczeniu: {total}")
    print(
        f"  Przeżyło:    {(df['Survived'] == 'survived').sum()} ({(df['Survived'] == 'survived').mean() * 100:.1f}%)"
    )
    print(
        f"  Nie przeżyło: {(df['Survived'] == 'not_survived').sum()} ({(df['Survived'] == 'not_survived').mean() * 100:.1f}%)"
    )
    print(f"  Kobiety: {(df['Sex'] == 'female').sum()}  Mężczyźni: {(df['Sex'] == 'male').sum()}")
    print(
        f"  1. klasa: {(df['Class'] == '1st').sum()}  2. klasa: {(df['Class'] == '2nd').sum()}  3. klasa: {(df['Class'] == '3rd').sum()}"
    )
    print()

    # 2. Apriori
    frequent_itemsets, _df_encoded = run_apriori(df)

    # 3. Reguły
    all_rules = extract_rules(frequent_itemsets)

    # 4. Ciekawe reguły
    survival_rules = filter_survival_rules(all_rules)

    # 5. Wykresy
    print("[plots] Generowanie wykresów...")
    plot_support_confidence(all_rules, survival_rules)
    plot_survival_bar(df)
    plot_heatmap_confidence(survival_rules)
    plot_lift_heatmap(survival_rules)

    # 6. Interpretacja
    print()
    print("=" * 70)
    print("INTERPRETACJA NAJCIEKAWSZYCH REGUŁ")
    print("=" * 70)
    print("""
  Na podstawie algorytmu Apriori (min_support=0.005, min_confidence=0.8):

  KLUCZOWE ZALEŻNOŚCI:
  ┌──────────────────────────────────────────────────────────┐
  │ 1. Kobiety z 1. i 2. klasy — najwyższe szanse przeżycia  │
  │    [Sex=female, Class=1st] → [Survived=survived]         │
  │    Ufność często > 0.95 (niemal pewna reguła)            │
  │                                                          │
  │ 2. Mężczyźni — niezależnie od klasy — rzadko przeżywali  │
  │    [Sex=male, Class=3rd] → [Survived=not_survived]       │
  │    Ufność > 0.85                                         │
  │                                                          │
  │ 3. Dzieci z 1. i 2. klasy — wysokie szanse przeżycia     │
  │    [Age=child, Class=1st] → [Survived=survived]          │
  │                                                          │
  │ 4. Pasażerowie 3. klasy — dominuje brak przeżycia        │
  │    [Class=3rd] → [Survived=not_survived]                 │
  └──────────────────────────────────────────────────────────┘

  WNIOSKI:
    • Zasada "kobiety i dzieci pierwsze" widoczna w danych empirycznie.
    • Klasa ekonomiczna miała istotny wpływ — bogaci pasażerowie mieli
      łatwiejszy dostęp do łodzi ratunkowych (górne pokłady).
    • Płeć jest silniejszym predykatorem przeżycia niż wiek.
    • Mężczyźni z 3. klasy mieli najniższe szanse przeżycia.
""")
    print(f"[done] Wykresy zapisane w: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
