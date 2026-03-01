import datetime
import math
import os

import matplotlib.pyplot as plt


PHYSICAL_CYCLE_DAYS = 23
EMOTIONAL_CYCLE_DAYS = 28
INTELLECTUAL_CYCLE_DAYS = 33

POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5
INTERSECTION_TOLERANCE = 0.05
MAX_SEARCH_DAYS = 365 * 125

PAST_WINDOW_DAYS = 30
FUTURE_WINDOW_DAYS = 60

OUTPUT_DIR = "lab01/biorhythms/output"
OUTPUT_FILE = "biorhythm_ai_gen_plot.png"


def prompt_non_empty(prompt_text: str) -> str:
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        print("Input cannot be empty.")


def prompt_birth_date() -> datetime.date:
    while True:
        value = input("What day were you born? (YYYY-MM-DD) ").strip()
        try:
            birth_date = datetime.date.fromisoformat(value)
        except ValueError:
            print("Incorrect date format, should be YYYY-MM-DD")
            continue

        if birth_date > datetime.date.today():
            print("Date of birth cannot be in the future.")
            continue

        return birth_date


def biorhythm_value(age_in_days: int, cycle_days: int) -> float:
    return math.sin((2 * math.pi / cycle_days) * age_in_days)


def cycle_message(cycle_name: str, today_value: float, tomorrow_value: float) -> None:
    print(f"{cycle_name} value today: {today_value:.3f}")

    if today_value >= POSITIVE_THRESHOLD:
        print(
            f"Great news: your {cycle_name.lower()} cycle is strong today. "
            "Use this momentum well!"
        )
    elif today_value <= NEGATIVE_THRESHOLD:
        print(
            f"Your {cycle_name.lower()} cycle is low today, but that is completely normal in biorhythms."
        )
        if tomorrow_value > today_value:
            print("Reassurance: tomorrow's value is higher, so things are trending up.")
        elif tomorrow_value < today_value:
            print("Reassurance: tomorrow's value is lower, but this dip is temporary and cyclical.")
        else:
            print("Reassurance: tomorrow's value is the same, and the cycle will keep moving soon.")
    else:
        print(f"Your {cycle_name.lower()} cycle is in a balanced range today.")

    print()


def find_closest_intersection_date(
    birth_date: datetime.date, today_date: datetime.date, search_days: int = MAX_SEARCH_DAYS
) -> tuple[datetime.date | None, float | None, int | None, int | None]:
    start_age_days = (today_date - birth_date).days

    best_score = float("inf")
    best_day = None
    best_target = None
    best_average = None

    for day in range(start_age_days, start_age_days + search_days + 1):
        p_val = biorhythm_value(day, PHYSICAL_CYCLE_DAYS)
        e_val = biorhythm_value(day, EMOTIONAL_CYCLE_DAYS)
        i_val = biorhythm_value(day, INTELLECTUAL_CYCLE_DAYS)

        spread = max(p_val, e_val, i_val) - min(p_val, e_val, i_val)
        avg_val = (p_val + e_val + i_val) / 3

        for target in (-1, 0, 1):
            target_distance = abs(avg_val - target)
            if spread <= INTERSECTION_TOLERANCE and target_distance <= INTERSECTION_TOLERANCE:
                score = spread + target_distance
                if score < best_score:
                    best_score = score
                    best_day = day
                    best_target = target
                    best_average = avg_val

    if best_day is None:
        return None, None, None, None

    event_date = birth_date + datetime.timedelta(days=best_day)
    age_years = best_day // 365
    return event_date, best_average, age_years, best_target


def plot_biorhythms(name: str, birth_date: datetime.date, today_date: datetime.date) -> str:
    start_date = today_date - datetime.timedelta(days=PAST_WINDOW_DAYS)
    end_date = today_date + datetime.timedelta(days=FUTURE_WINDOW_DAYS)

    dates = []
    physical_values = []
    emotional_values = []
    intellectual_values = []

    current_date = start_date
    while current_date <= end_date:
        age_in_days = (current_date - birth_date).days
        dates.append(current_date)
        physical_values.append(biorhythm_value(age_in_days, PHYSICAL_CYCLE_DAYS))
        emotional_values.append(biorhythm_value(age_in_days, EMOTIONAL_CYCLE_DAYS))
        intellectual_values.append(biorhythm_value(age_in_days, INTELLECTUAL_CYCLE_DAYS))
        current_date += datetime.timedelta(days=1)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, physical_values, label="Physical", linewidth=2)
    plt.plot(dates, emotional_values, label="Emotional", linewidth=2)
    plt.plot(dates, intellectual_values, label="Intellectual", linewidth=2)

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.axvline(today_date, color="blue", linestyle="--", linewidth=1.2, label="Today")

    if start_date <= birth_date <= end_date:
        plt.axvline(
            birth_date,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"Birthday ({birth_date.isoformat()})",
        )

    plt.title(f"Biorhythms for {name} ({PAST_WINDOW_DAYS} days back, {FUTURE_WINDOW_DAYS} days ahead)")
    plt.xlabel("Date")
    plt.ylabel("Cycle value")
    plt.ylim(-1.1, 1.1)
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=300)

    if "agg" not in plt.get_backend().lower():
        plt.show()

    return output_path


def main() -> None:
    print("Hi, give me your name and date of birth")
    forename = prompt_non_empty("What is your first name? ")
    surname = prompt_non_empty("What is your last name? ")
    full_name = f"{forename} {surname}"
    birth_date = prompt_birth_date()

    print()
    print(f"Okay, buckle up {full_name}! Time to calculate your biorhythms.")
    print()

    today = datetime.date.today()
    age_today_days = (today - birth_date).days

    print(f"Your age in days is {age_today_days}")
    print()

    physical_today = biorhythm_value(age_today_days, PHYSICAL_CYCLE_DAYS)
    physical_tomorrow = biorhythm_value(age_today_days + 1, PHYSICAL_CYCLE_DAYS)
    cycle_message("Physical", physical_today, physical_tomorrow)

    emotional_today = biorhythm_value(age_today_days, EMOTIONAL_CYCLE_DAYS)
    emotional_tomorrow = biorhythm_value(age_today_days + 1, EMOTIONAL_CYCLE_DAYS)
    cycle_message("Emotional", emotional_today, emotional_tomorrow)

    intellectual_today = biorhythm_value(age_today_days, INTELLECTUAL_CYCLE_DAYS)
    intellectual_tomorrow = biorhythm_value(age_today_days + 1, INTELLECTUAL_CYCLE_DAYS)
    cycle_message("Intellectual", intellectual_today, intellectual_tomorrow)

    event_date, avg_value, age_years, target = find_closest_intersection_date(birth_date, today)
    print("=== Trivia ===")
    if event_date is None or avg_value is None or age_years is None or target is None:
        print("No near-intersection found within the search window using 5% tolerance.")
    else:
        print(
            "Closest near-intersection date (within 5% tolerance): "
            f"{event_date.isoformat()}"
        )
        print(f"The cycles are clustering near target value {target} with average {avg_value:.3f}.")
        print(f"Predicted age at that time: approximately {age_years} years old.")

    plot_path = plot_biorhythms(full_name, birth_date, today)
    print()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
