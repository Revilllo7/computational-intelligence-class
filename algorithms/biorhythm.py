from __future__ import annotations

import math
from typing import Iterable

DEFAULT_CYCLES: tuple[int, int, int] = (23, 28, 33)


def biorhythm_value(age_days: int, cycle_days: int) -> float:
    # Return biorhythm value in range [-1, 1] for age and cycle length.
    return math.sin((2 * math.pi / cycle_days) * age_days)


def biorhythm_triplet(
    age_days: int,
    cycles: tuple[int, int, int] = DEFAULT_CYCLES,
) -> tuple[float, float, float]:
    # Return (physical, emotional, intellectual) biorhythm values.
    physical = biorhythm_value(age_days, cycles[0])
    emotional = biorhythm_value(age_days, cycles[1])
    intellectual = biorhythm_value(age_days, cycles[2])
    return physical, emotional, intellectual


def generate_cycle_series(
    start_day: int,
    end_day: int,
    cycle_days: int,
) -> list[float]:
    # Generate biorhythm values for inclusive day range.
    return [biorhythm_value(day, cycle_days) for day in range(start_day, end_day + 1)]


def find_next_intersection(
    start_day: int,
    cycles: Iterable[int] = DEFAULT_CYCLES,
    max_search_days: int = 45656,
    intersection_tolerance: float = 0.01,
    target_tolerance: float = 0.05,
) -> tuple[int | None, float | None]:
    # Find next day where all cycles are close together and near -1, 0, or 1.
    cycle_tuple = tuple(cycles)

    for day in range(start_day + 1, start_day + max_search_days):
        values = [biorhythm_value(day, cycle_days) for cycle_days in cycle_tuple]
        if max(values) - min(values) >= intersection_tolerance:
            continue

        average_value = sum(values) / len(values)
        if (
            abs(average_value - 1.0) < target_tolerance
            or abs(average_value) < target_tolerance
            or abs(average_value + 1.0) < target_tolerance
        ):
            return day, average_value

    return None, None
