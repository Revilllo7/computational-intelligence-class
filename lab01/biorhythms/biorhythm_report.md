# Lab01

## Task01: Biorhythms

`biorhythm.py` is a Python script to calculate and plot user biorhythms and give information about the current phase they are in.

***Usage***:
```bash
python biorhythm.py
```
Afterwards answer the prompts with information.

### Output
The console will give you information regarding your phase and the plot will be saved in `lab01/biorhythms/output/biorhythm_plot.png` with a visual representation of your biorhythms for 30 days prior and 60 days ahead.

### Console output example
```yaml
Hi, give me your name and date of birth
What is your first name? John
What is your last name? Doe
What day were you born? (YYYY-MM-DD) 1990-01-01

Okay, buckle up John Doe it's time for the maths!

Your age in days is 13208

physical_biometric = 0.9976687691905489
Your physical biorhythm is good!

emotional_biometric = -0.9749279121818011
Your emotional biorhythm is bad...
And it will be worse tomorrow :)

intellectual_biometric = 0.9988673391830092
Your intellectual biorhythm is good!

All three biorhythms will intersect at approximately 0.975 on day 17002
That will be on: 2036-07-20
You will be 46 years and 7 months old at that time.

Plot saved to lab01/biorhythms/output/biorhythm_plot.png
```

### Plot output examples

**Biorhythm plot for 2026-02-15**
![biorhythm-2026](./markdown/biorhythm_2026.png)
> the only biorhythm that shows the marked birthday as it's withing the visible range of -30 days.

**Biorhythm plot for 2013-11-17**
![biorhythm-2013](./markdown/biorhythm_2013.png)

**Biorhythm plot for 2000-01-01**
![biorhythm-2000](./markdown/biorhythm_2000.png)

### Math explanation

The graphs intersect at very specific points that's why multiple users will share their closest intersection point. I tried to graph it nicely, but failed horribly, so here's your link to desmos and you're free to scroll to the x (day value) and compare:
`https://www.desmos.com/calculator/7qzzhhujeu`

### Report information:
Time taken (rough estimates):
20 minutes to make the structure and put the plans on the board
25 minutes to write the code and figure out the maths (ignoring the plotting part)
60? minutes to make the plotting logic and formatting
15 minutes to write the report and make the screenshots