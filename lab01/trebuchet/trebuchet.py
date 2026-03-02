import math
import os
from random import randint
import matplotlib.pyplot as plt

HEIGHT = 100 # meters
INITIAL_VELOCITY = 50 # m/s
GRAVITY = 9.81 # m/s^2
SHOT_RANGE = [50, 340] # meters
ERROR_MARGIN = 5 # meters
DEGREES_RANGE = [0, 90] # degrees

OUTPUT_DIR = "trebuchet/output"
OUTPUT_FILE = "trebuchet_plot.png"


# PROMPTING THE USER FOR INPUT

def prompt_non_empty() -> int:
    while True:
        value = input(f"Enter the angle in degrees (must be between {DEGREES_RANGE[0]} and {DEGREES_RANGE[1]}): ").strip()
        if value:
            try:
                angle = int(value)
                if DEGREES_RANGE[0] <= angle <= DEGREES_RANGE[1]:
                    return angle
                else:
                    print(f"Angle must be between {DEGREES_RANGE[0]} and {DEGREES_RANGE[1]}.")
            except ValueError:
                print("Input must be an integer.")
        print("Input cannot be empty.")

def calculate_elevated_range(angle_degrees: int) -> float:
    angle_rad = math.radians(angle_degrees)
    return (INITIAL_VELOCITY * math.cos(angle_rad)) * ((INITIAL_VELOCITY * math.sin(angle_rad) + math.sqrt((INITIAL_VELOCITY * math.sin(angle_rad))**2 + 2 * GRAVITY * HEIGHT)) / GRAVITY)
    # distance = initial velocity * cos(angle) * ((initial velocity * sin(angle) + sqrt((initial velocity * sin(angle))^2 + 2 * gravity * height)) / gravity)



# CALCULATING TRAJECTORY

def calculate_trajectory(angle_degrees: int) -> tuple:
    # Calculate projectile trajectory x, y coordinates for given angle
    angle_rad = math.radians(angle_degrees)
    
    # Calculate time when projectile hits ground (y = 0) using quadratic formula
    # 0.5 * g * t^2 - v*sin(angle) * t - HEIGHT = 0
    a = 0.5 * GRAVITY
    b = -INITIAL_VELOCITY * math.sin(angle_rad)
    c = -HEIGHT
    
    discriminant = b**2 - 4*a*c
    t_impact = (-b + math.sqrt(discriminant)) / (2*a)
    
    # Generate time array
    time_steps = 100
    time_array = [i * t_impact / time_steps for i in range(time_steps + 1)]

    # INFO:
    # There's a formula to calculate it that ignores the time:
    # Y = -(g/2 * v^2 * cos^2(angle)) * X^2 + tan(angle) * X + HEIGHT
    # But I feel like brute forcing the time is easier for me to understand

    # Calculate x, y coordinates
    x_coords = [INITIAL_VELOCITY * math.cos(angle_rad) * time_interval for time_interval in time_array]
    y_coords = [HEIGHT + INITIAL_VELOCITY * math.sin(angle_rad) * time_interval - 0.5 * GRAVITY * time_interval**2 for time_interval in time_array]
    
    return x_coords, y_coords



# TARGET SPAWNING

def spawn_target() -> int:
    return randint(SHOT_RANGE[0], SHOT_RANGE[1])



# PLOTTING RESULTS

def plot_results(shot_history: list, target: int) -> None:
    # Plot projectile trajectories with winning shot highlighted and previous transparent
    plt.figure(figsize=(12, 7))
    
    # Calculate all trajectories and track max height
    all_trajectories = []
    max_height = 0
    
    # Plot previous shots with lower opacity
    for shot in shot_history[:-1]:
        x_coords, y_coords = calculate_trajectory(shot['angle'])
        all_trajectories.append((x_coords, y_coords))
        max_height = max(max_height, max(y_coords))
        plt.plot(x_coords, y_coords, color='blue', linestyle='-', linewidth=1.5, alpha=0.25)
    
    # Plot winning shot
    winning_shot = shot_history[-1]
    x_coords, y_coords = calculate_trajectory(winning_shot['angle'])
    max_height = max(max_height, max(y_coords))
    plt.plot(x_coords, y_coords, color='blue', linestyle='-', linewidth=2.5, alpha=0.8, label='Trajectory of projectile')
    
    # Mark the hit point
    plt.plot(winning_shot['distance'], 0, 'o', markersize=20, label='Hit point', zorder=5)
    
    # Plot target line
    plt.axvline(x=target, color='red', linestyle='--', linewidth=2.5, label=f'Distance d = {winning_shot["distance"]:.2f} m')
    
    # Plot ground
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Annotations
    plt.text(20, HEIGHT - 5, f'α = {winning_shot["angle"]}°', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    plt.text(20, HEIGHT - 20, f'Initial velocity v₀ = {INITIAL_VELOCITY}m/s', fontsize=11, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    # Labels and title
    plt.xlabel('Distance (m)', fontsize=12)
    plt.ylabel('Height (m)', fontsize=12)
    plt.title('Projectile Motion for the Trebuchet', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.xlim(0, max([s['distance'] for s in shot_history] + [target]) + 50)
    plt.ylim(0, max_height + 20)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")



# MAIN GAME LOOP

def main():
    # Spawn target first
    target = spawn_target()
    print(f"\n{'='*50}")
    print(f"TARGET DISTANCE: {target} meters")
    print(f"{'='*50}\n")
    
    shot_history = []
    shot_count = 0
    
    # Keep prompting until target is hit
    while True:
        shot_count += 1
        print(f"\n--- Attempt {shot_count} ---")
        angle = prompt_non_empty()
        distance = calculate_elevated_range(angle)
        error = distance - target
        
        print(f"Shot distance: {distance:.2f} meters")
        
        # Check if hit
        if abs(error) <= ERROR_MARGIN:
            print(f"\nHIT! Distance error: {error:+.2f} meters (within {ERROR_MARGIN}m)")
            shot_history.append({'angle': angle, 'distance': distance, 'hit': True})
            break
        else:
            shot_history.append({'angle': angle, 'distance': distance, 'hit': False})
            # Keep only last 5 shots
            if len(shot_history) > 5:
                shot_history.pop(0)
            print(f"Miss! Distance error: {error:+.2f} meters")
    
    # Display final results and plot
    print(f"\n{'='*50}")
    print(f"SUCCESS! Hit target in {shot_count} attempt(s)")
    print(f"Final angle: {angle}°")
    print(f"{'='*50}\n")
    
    plot_results(shot_history, target)

if __name__ == "__main__":
    main()