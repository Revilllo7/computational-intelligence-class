# A program to calculate the user's biorhythms: physical, emotional, intellectual

import datetime
import math
import os
import matplotlib.pyplot as plt

print("Hi, give me your name and date of birth")
forename = str(input("What is your first name? "))
surname = str(input("What is your last name? "))
name = forename + " " + surname
date_of_birth = str(input("What day were you born? (YYYY-MM-DD) "))

def date_validation(string):
    try:
        datetime.date.fromisoformat(string)
    except:
        print("Incorrect data format, should be YYYY-MM-DD")
        # new_date_of_birth = str(input("What day were you born? (YYYY-MM-DD) "))

date_validation(date_of_birth)
print()
print("Okay, buckle up " + name + " it's time for the maths!")
print()
current_date = datetime.date.today().isoformat()

age_in_days = (datetime.date.fromisoformat(current_date) - datetime.date.fromisoformat(date_of_birth)).days

print("Your age in days is " + str(age_in_days))
print()
# BIOMETRY
# if closer to 1 -> good
# if closer to 0 -> neutral
# if closer to -1 -> bad

# PHYSICAL BIOMETRY
# Yp = sin((2pi / 23)*t)
physical_biometric = math.sin((2 * math.pi / 23) * age_in_days)
print(f"physical_biometric = {physical_biometric}")
if physical_biometric >= 0.5:
    print("Your physical biorhythm is good!")
elif physical_biometric <= -0.5:
    print("Your physical biorhythm is bad...")
    physical_biometric_next_day = math.sin((2 * math.pi / 23) * (age_in_days + 1))
    if physical_biometric_next_day > physical_biometric:
        print("But it will be better tomorrow!")
    else:
        print("And it will be worse tomorrow :)")
else:
    print("Your physical biorhythm is neutral.")
print()

# EMOTIONAL BIOMETRY
# Ye = sin((2pi / 28)*t)
emotional_biometric = math.sin((2 * math.pi / 28) * age_in_days)
print(f"emotional_biometric = {emotional_biometric}")
if emotional_biometric >= 0.5:
    print("Your emotional biorhythm is good!")
elif emotional_biometric <= -0.5:
    print("Your emotional biorhythm is bad...")
    emotional_biometric_next_day = math.sin((2 * math.pi / 28) * (age_in_days + 1))
    if emotional_biometric_next_day > emotional_biometric:
        print("But it will be better tomorrow!")
    else:
        print("And it will be worse tomorrow :)")
else:
    print("Your emotional biorhythm is neutral.")
print()

# INTELLECTUAL BIOMETRY
# Yi = sin((2pi / 33)*t)
intellectual_biometric = math.sin((2 * math.pi / 33) * age_in_days)
print(f"intellectual_biometric = {intellectual_biometric}")
if intellectual_biometric >= 0.5:
    print("Your intellectual biorhythm is good!")
elif intellectual_biometric <= -0.5:
    print("Your intellectual biorhythm is bad...")
    intellectual_biometric_next_day = math.sin((2 * math.pi / 33) * (age_in_days + 1))
    if intellectual_biometric_next_day > intellectual_biometric:
        print("But it will be better tomorrow!")
    else:
        print("And it will be worse tomorrow :)")
else:
    print("Your intellectual biorhythm is neutral.")
print()



# PLOT FUNCTIONALITY

# Find the next intersection point at -1, 0, or 1 after today
def find_next_intersection(start_day, max_search_days=45656): # I took the oldest person ever alive as reference and rounded up to 125 years
    # Find the next day where all three biorhythms intersect at -1, 0, or 1
    tolerance = 0.01  # How close values need to be to each other
    target_tolerance = 0.05  # How close to -1, 0, or 1 (-0.95 to 0.95)
    
    for day in range(start_day + 1, start_day + max_search_days):
        physical = math.sin((2 * math.pi / 23) * day)
        emotional = math.sin((2 * math.pi / 28) * day)
        intellectual = math.sin((2 * math.pi / 33) * day)
        
        # Check if all three are close to each other
        if abs(physical - emotional) < tolerance and abs(emotional - intellectual) < tolerance and abs(physical - intellectual) < tolerance:
            # Check if they're close to -1, 0, or 1
            avg_val = (physical + emotional + intellectual) / 3
            if abs(avg_val - 1) < target_tolerance or abs(avg_val) < target_tolerance or abs(avg_val + 1) < target_tolerance:
                return day, avg_val
    
    return None, None

intersection_day, intersection_value = find_next_intersection(age_in_days)

if intersection_day and intersection_value is not None:
    # Format intersection value to avoid -0.000
    formatted_value = abs(intersection_value) if abs(intersection_value) < 0.01 else intersection_value
    print(f"All three biorhythms will intersect at approximately {formatted_value:.3f} on day {intersection_day}")
    intersection_date = datetime.date.fromisoformat(date_of_birth) + datetime.timedelta(days=intersection_day)
    print(f"That will be on: {intersection_date.isoformat()}")
    print(f"You will be {intersection_day // 365} years and {(intersection_day % 365) // 30} months old at that time.")
else:
    print("No intersection found within search range.")

# Generate data from 30 days before today to 60 days in the future
start_day = max(0, age_in_days - 30)  # Don't go before birth
end_day = age_in_days + 60
days = list(range(start_day, end_day + 1))
physical_biometric_values = [math.sin((2 * math.pi / 23) * day) for day in days]
emotional_biometric_values = [math.sin((2 * math.pi / 28) * day) for day in days]
intellectual_biometric_values = [math.sin((2 * math.pi / 33) * day) for day in days]

# PLOTTING

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(days, physical_biometric_values, label='Physical', alpha=0.8)
plt.plot(days, emotional_biometric_values, label='Emotional', alpha=0.8)
plt.plot(days, intellectual_biometric_values, label='Intellectual', alpha=0.8)
plt.axhline(0, color='black', lw=0.5, ls='--')

# Mark birthday (day 0) if it's in the visible range (biorhythm-2026 example)
if start_day == 0:
    plt.axvline(0, color='red', lw=2.5, ls='--', label=f'Birthday ({date_of_birth})')

# Mark today
today_date = datetime.date.fromisoformat(current_date)
plt.axvline(age_in_days, color='blue', lw=1.5, ls='--', label=f'Today (day {age_in_days}, {current_date})')

# Mark tomorrow
tomorrow_date = today_date + datetime.timedelta(days=1)
plt.axvline(age_in_days + 1, color='cyan', lw=1, ls=':', label=f'Tomorrow (day {age_in_days + 1}, {tomorrow_date.isoformat()})')

# Set x-axis limits to only show the relevant range
plt.xlim(start_day, end_day)

plt.title(f'Biorhythms for {name} (30 days prior, 60 days ahead)')
plt.xlabel('Days from birth')
plt.ylabel('Value')
plt.legend(loc='best', fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)
plt.savefig('output/biorhythm_plot.png', dpi=600)
print()
print("Plot saved to output/biorhythm_plot.png")
