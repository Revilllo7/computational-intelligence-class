import datetime
import math
import os

import matplotlib.pyplot as plt


PHYSICAL_CYCLE_DAYS = 23
EMOTIONAL_CYCLE_DAYS = 28
INTELLECTUAL_CYCLE_DAYS = 33

NEGATIVE_THRESHOLD = -0.5
POSITIVE_THRESHOLD = 0.5

INTERSECTION_TOLERANCE = 0.01
TARGET_TOLERANCE = 0.05
MAX_INTERSECTION_SEARCH_DAYS = 45656

PAST_WINDOW_DAYS = 30
FUTURE_WINDOW_DAYS = 60

OUTPUT_DIR = "output"
OUTPUT_FILE = "biorhythm_plot.png"


def prompt_non_empty(prompt: str) -> str:
	while True:
		value = input(prompt).strip()
		if value:
			return value
		print("Input cannot be empty.")


def prompt_birth_date() -> datetime.date:
	while True:
		raw = input("What day were you born? (YYYY-MM-DD) ").strip()
		try:
			birth_date = datetime.date.fromisoformat(raw)
		except ValueError:
			print("Incorrect date format, should be YYYY-MM-DD")
			continue

		if birth_date > datetime.date.today():
			print("Date of birth cannot be in the future.")
			continue

		return birth_date


def biorhythm_value(age_days: int, cycle_days: int) -> float:
	return math.sin((2 * math.pi / cycle_days) * age_days)


def describe_biorhythm(name: str, value: float, next_day_value: float) -> None:
	print(f"{name}_biorhythm = {value}")
	if value >= POSITIVE_THRESHOLD:
		print(f"Your {name} biorhythm is good!")
	elif value <= NEGATIVE_THRESHOLD:
		print(f"Your {name} biorhythm is bad...")
		if next_day_value > value:
			print("But it will be better tomorrow!")
		else:
			print("And it will be worse tomorrow :)")
	else:
		print(f"Your {name} biorhythm is neutral.")
	print()


def find_next_intersection(start_day: int) -> tuple[int | None, float | None]:
	for day in range(start_day + 1, start_day + MAX_INTERSECTION_SEARCH_DAYS):
		physical = biorhythm_value(day, PHYSICAL_CYCLE_DAYS)
		emotional = biorhythm_value(day, EMOTIONAL_CYCLE_DAYS)
		intellectual = biorhythm_value(day, INTELLECTUAL_CYCLE_DAYS)

		close_together = (
			abs(physical - emotional) < INTERSECTION_TOLERANCE
			and abs(emotional - intellectual) < INTERSECTION_TOLERANCE
			and abs(physical - intellectual) < INTERSECTION_TOLERANCE
		)

		if not close_together:
			continue

		average_value = (physical + emotional + intellectual) / 3
		close_to_target = (
			abs(average_value - 1) < TARGET_TOLERANCE
			or abs(average_value) < TARGET_TOLERANCE
			or abs(average_value + 1) < TARGET_TOLERANCE
		)

		if close_to_target:
			return day, average_value

	return None, None


def safe_zero(value: float, epsilon: float = 0.01) -> float:
	return 0.0 if abs(value) < epsilon else value


def main() -> None:
	print("Hi, give me your name and date of birth")
	forename = prompt_non_empty("What is your first name? ")
	surname = prompt_non_empty("What is your last name? ")
	name = f"{forename} {surname}"
	birth_date = prompt_birth_date()

	print()
	print(f"Okay, buckle up {name} it's time for the maths!")
	print()

	today_date = datetime.date.today()
	age_in_days = (today_date - birth_date).days
	print(f"Your age in days is {age_in_days}")
	print()

	physical_today = biorhythm_value(age_in_days, PHYSICAL_CYCLE_DAYS)
	physical_tomorrow = biorhythm_value(age_in_days + 1, PHYSICAL_CYCLE_DAYS)
	describe_biorhythm("physical", physical_today, physical_tomorrow)

	emotional_today = biorhythm_value(age_in_days, EMOTIONAL_CYCLE_DAYS)
	emotional_tomorrow = biorhythm_value(age_in_days + 1, EMOTIONAL_CYCLE_DAYS)
	describe_biorhythm("emotional", emotional_today, emotional_tomorrow)

	intellectual_today = biorhythm_value(age_in_days, INTELLECTUAL_CYCLE_DAYS)
	intellectual_tomorrow = biorhythm_value(age_in_days + 1, INTELLECTUAL_CYCLE_DAYS)
	describe_biorhythm("intellectual", intellectual_today, intellectual_tomorrow)

	intersection_day, intersection_value = find_next_intersection(age_in_days)
	if intersection_day is not None and intersection_value is not None:
		display_value = safe_zero(intersection_value)
		intersection_date = birth_date + datetime.timedelta(days=intersection_day)
		years_old = intersection_day // 365

		print("=== Fun Trivia ===")
		print(
			f"All three biorhythms will intersect at approximately {display_value:.3f} "
			f"on day {intersection_day}"
		)
		print(f"That will be on: {intersection_date.isoformat()}")
		print(f"You will be approximately {years_old} years old at that time.")
	else:
		print("=== Fun Trivia ===")
		print("No intersection found within search range.")

	start_day = max(0, age_in_days - PAST_WINDOW_DAYS)
	end_day = age_in_days + FUTURE_WINDOW_DAYS
	days = list(range(start_day, end_day + 1))

	physical_values = [biorhythm_value(day, PHYSICAL_CYCLE_DAYS) for day in days]
	emotional_values = [biorhythm_value(day, EMOTIONAL_CYCLE_DAYS) for day in days]
	intellectual_values = [biorhythm_value(day, INTELLECTUAL_CYCLE_DAYS) for day in days]

	plt.figure(figsize=(12, 6))
	plt.plot(days, physical_values, label="Physical", alpha=0.8)
	plt.plot(days, emotional_values, label="Emotional", alpha=0.8)
	plt.plot(days, intellectual_values, label="Intellectual", alpha=0.8)
	plt.axhline(0, color="black", lw=0.5, ls="--")

	if start_day == 0:
		plt.axvline(0, color="red", lw=2.5, ls="--", label=f"Birthday ({birth_date.isoformat()})")

	tomorrow_date = today_date + datetime.timedelta(days=1)
	plt.axvline(
		age_in_days,
		color="blue",
		lw=1.5,
		ls="--",
		label=f"Today (day {age_in_days}, {today_date.isoformat()})",
	)
	plt.axvline(
		age_in_days + 1,
		color="cyan",
		lw=1,
		ls=":",
		label=f"Tomorrow (day {age_in_days + 1}, {tomorrow_date.isoformat()})",
	)

	plt.xlim(start_day, end_day)
	plt.title(f"Biorhythms for {name} (30 days prior, 60 days ahead)")
	plt.xlabel("Days from birth")
	plt.ylabel("Value")
	plt.legend(loc="best", fontsize=8)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	os.makedirs(OUTPUT_DIR, exist_ok=True)
	output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
	plt.savefig(output_path, dpi=300)

	print()
	print(f"Plot saved to {output_path}")

	if "agg" not in plt.get_backend().lower():
		plt.show()


if __name__ == "__main__":
	main()
