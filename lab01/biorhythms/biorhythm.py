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
print(physical_biometric)
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
print(emotional_biometric)
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
print(intellectual_biometric)
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


