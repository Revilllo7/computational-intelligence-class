"""
Maze Definition for Task 03
============================

12x12 maze with borders (10x10 internal grid)
0 = passable
1 = wall

Format:
- First row/column: border
- Last row/column: border
- Interior: 10x10 grid with walls
"""

# 12x12 maze (including borders)
# Start: (1, 1) - top-left corner of internal grid
# Goal: (10, 10) - bottom-right corner of internal grid
# Maximum steps allowed: 30 (actual solution ~20)

maze = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Coordinates
START = (1, 1)      # Top-left of internal grid
GOAL = (10, 10)     # Bottom-right of internal grid
MAX_STEPS = 30      # Maximum allowed steps

def print_maze():
    """Print maze for visualization"""
    for row in maze:
        print(''.join(['█' if cell == 1 else ' ' for cell in row]))

def is_passable(x: int, y: int) -> bool:
    """Check if a cell is passable (within bounds and not a wall)"""
    if 0 <= x < 12 and 0 <= y < 12:
        return maze[y][x] == 0
    return False

if __name__ == "__main__":
    print("Maze:")
    print_maze()
    print(f"\nStart: {START}")
    print(f"Goal: {GOAL}")
    print(f"Max steps: {MAX_STEPS}")
