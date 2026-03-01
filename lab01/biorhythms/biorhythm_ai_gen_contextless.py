import os
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt

# Biorhythm periods
PHYSICAL = 23
EMOTIONAL = 28
INTELLECTUAL = 33

TOLERANCE = 0.05  # 5% tolerance for intersections


def calculate_days_lived(dob, target_date):
    return (target_date - dob).days


def biorhythm_value(days, period):
    return math.sin(2 * math.pi * days / period)


def motivate_or_reassure(name, cycle_name, today_val, tomorrow_val):
    if today_val > 0.5:
        print(f"💪 {name}, your {cycle_name} cycle is high today ({today_val:.2f}). Great time to shine!")
    elif today_val < -0.5:
        direction = "higher" if tomorrow_val > today_val else "lower"
        print(f"⚠️ {name}, your {cycle_name} cycle is low today ({today_val:.2f}). "
              f"Tomorrow will be {direction} ({tomorrow_val:.2f}). Take care!")
    else:
        print(f"🙂 {name}, your {cycle_name} cycle is balanced today ({today_val:.2f}). Stay steady!")


def find_closest_intersection(dob, start_date, end_date):
    best_date = None
    best_score = float('inf')

    targets = [-1, 0, 1]

    current_date = start_date
    while current_date <= end_date:
        days = calculate_days_lived(dob, current_date)

        p = biorhythm_value(days, PHYSICAL)
        e = biorhythm_value(days, EMOTIONAL)
        i = biorhythm_value(days, INTELLECTUAL)

        for t in targets:
            if (abs(p - t) <= TOLERANCE and
                abs(e - t) <= TOLERANCE and
                abs(i - t) <= TOLERANCE):

                score = abs(p - t) + abs(e - t) + abs(i - t)
                if score < best_score:
                    best_score = score
                    best_date = current_date

        current_date += timedelta(days=1)

    return best_date


def main():
    # User input
    name = input("Enter your name: ").strip()
    dob_str = input("Enter your date of birth (YYYY-MM-DD): ").strip()

    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    today = datetime.today().date()
    tomorrow = today + timedelta(days=1)

    # Calculate today's and tomorrow's cycles
    days_today = calculate_days_lived(dob, today)
    days_tomorrow = calculate_days_lived(dob, tomorrow)

    p_today = biorhythm_value(days_today, PHYSICAL)
    e_today = biorhythm_value(days_today, EMOTIONAL)
    i_today = biorhythm_value(days_today, INTELLECTUAL)

    p_tomorrow = biorhythm_value(days_tomorrow, PHYSICAL)
    e_tomorrow = biorhythm_value(days_tomorrow, EMOTIONAL)
    i_tomorrow = biorhythm_value(days_tomorrow, INTELLECTUAL)

    print("\n--- Today's Biorhythm ---")
    print(f"Physical:     {p_today:.2f}")
    print(f"Emotional:    {e_today:.2f}")
    print(f"Intellectual: {i_today:.2f}\n")

    # Motivational / reassurance messages
    motivate_or_reassure(name, "physical", p_today, p_tomorrow)
    motivate_or_reassure(name, "emotional", e_today, e_tomorrow)
    motivate_or_reassure(name, "intellectual", i_today, i_tomorrow)

    # Plot range
    start_date = today - timedelta(days=30)
    end_date = today + timedelta(days=60)

    dates = []
    physical_vals = []
    emotional_vals = []
    intellectual_vals = []

    current_date = start_date
    while current_date <= end_date:
        days = calculate_days_lived(dob, current_date)
        dates.append(current_date)
        physical_vals.append(biorhythm_value(days, PHYSICAL))
        emotional_vals.append(biorhythm_value(days, EMOTIONAL))
        intellectual_vals.append(biorhythm_value(days, INTELLECTUAL))
        current_date += timedelta(days=1)

    # Create output directories
    output_dir = os.path.join(os.getcwd(), "biorhythms", "output")
    os.makedirs(output_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, physical_vals, label="Physical Cycle")
    plt.plot(dates, emotional_vals, label="Emotional Cycle")
    plt.plot(dates, intellectual_vals, label="Intellectual Cycle")

    # Mark birthday if visible (for infants)
    if start_date <= dob <= end_date:
        plt.axvline(dob, linestyle="--", label="Birthday")

    plt.title(f"{name}'s Biorhythm Cycles")
    plt.xlabel("Date")
    plt.ylabel("Cycle Value")
    plt.legend()
    plt.grid(True)

    filename = f"{name.lower()}_biorhythm.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    print(f"\nPlot saved to: {filepath}")

    # Trivia: closest intersection
    intersection_date = find_closest_intersection(dob, today, today + timedelta(days=365 * 5))

    if intersection_date:
        age = (intersection_date - dob).days // 365
        print("\n--- Trivia ---")
        print(f"Closest cycle intersection occurs on: {intersection_date}")
        print(f"Predicted age on that date: {age} years")
    else:
        print("\n--- Trivia ---")
        print("No close intersection found within the searched range.")


if __name__ == "__main__":
    main()