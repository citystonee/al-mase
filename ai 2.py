import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Constants for maze dimensions
WIDTH = 41  # Must be odd numbers
HEIGHT = 41  # Must be odd numbers

# Directions
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_EFFECTS = {
    'up': (0, -1),
    'down': (0, 1),
    'left': (-1, 0),
    'right': (1, 0)
}

def generate_maze(width, height):
    # Initialize the maze grid with walls (1)
    maze = np.ones((height, width), dtype=int)

    N, S, E, W = ('N', 'S', 'E', 'W')
    DX = {E: 2, W: -2, N: 0, S: 0}
    DY = {E: 0, W: 0, N: -2, S: 2}

    def carve_passages(cx, cy):
        directions = [N, S, E, W]
        random.shuffle(directions)
        for direction in directions:
            nx, ny = cx + DX[direction], cy + DY[direction]
            if 0 < nx < width and 0 < ny < height and maze[ny][nx] == 1:
                maze[cy + DY[direction] // 2][cx + DX[direction] // 2] = 0  # Remove wall between cells
                maze[ny][nx] = 0  # Mark cell as passage
                carve_passages(nx, ny)

    # Start carving from (1,1)
    maze[1][1] = 0
    carve_passages(1, 1)

    return maze

class Agent:
    def __init__(self, maze, start, end, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, visualize=False, visualize_interval=10):
        self.maze = maze
        self.start = start
        self.end = end
        self.position = start
        self.q_table = {}
        self.episodes = episodes
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.visualize = visualize  # Whether to visualize during learning
        self.visualize_interval = visualize_interval  # Visualize every N episodes
        self.build_q_table()


    def take_action(self, state, action):
        x, y = state
        dx, dy = ACTION_EFFECTS[action]
        nx, ny = x + dx, y + dy
        if (0 <= nx < self.maze.shape[1] and 0 <= ny < self.maze.shape[0] and self.maze[ny][nx] == 0):
            next_state = (nx, ny)
            if next_state == self.end:
                reward = 100  # High reward for reaching the goal
            else:
                reward = -1  # Small penalty for each move
        else:
            next_state = state
            reward = -10  # Penalty for hitting a wall
        return next_state, reward

    # (rest of the code remains the same)


    def build_q_table(self):
        # Initialize Q-table with zeros for all state-action pairs
        for y in range(self.maze.shape[0]):
            for x in range(self.maze.shape[1]):
                if self.maze[y][x] == 0:
                    self.q_table[(x, y)] = {action: 0.0 for action in ACTIONS}

    def choose_action(self, state):
        # Epsilon-greedy strategy
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            # Choose action with highest Q-value
            q_values = self.q_table[state]
            max_value = max(q_values.values())
            max_actions = [action for action, value in q_values.items() if value == max_value]
            action = random.choice(max_actions)
        return action

    def learn(self):
        for episode in range(1, self.episodes + 1):
            self.alpha = max(0.1, self.alpha * 0.995)  # Gradually decrease learning rate
            self.epsilon = max(0.1, self.epsilon * 0.995)  # Gradually decrease epsilon
            state = self.start
            self.position = state
            steps = 0
            episode_rewards = 0
            # (rest of the code remains the same)

                # For visualization
            if self.visualize and episode % self.visualize_interval == 0:
                plt.ion()
                fig, ax = plt.subplots(figsize=(8, 8))

            while state != self.end:
                action = self.choose_action(state)
                next_state, reward = self.take_action(state, action)
                # Update Q-value
                old_value = self.q_table[state][action]
                next_max = max(self.q_table[next_state].values())
                new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                self.q_table[state][action] = new_value

                state = next_state
                steps += 1
                episode_rewards += reward

                # Visualization during learning
                if self.visualize and episode % self.visualize_interval == 0:
                    ax.clear()
                    ax.imshow(self.maze, cmap='binary', interpolation='nearest')
                    ax.scatter(state[0], state[1], color='blue', s=100)
                    ax.scatter(self.start[0], self.start[1], color='green', s=100)  # Start
                    ax.scatter(self.end[0], self.end[1], color='red', s=100)  # End
                    ax.set_xticks([]), ax.set_yticks([])
                    ax.set_title(f'Episode {episode}/{self.episodes}')
                    plt.pause(0.01)

                if steps > 1000:
                    break  # Prevent infinite loops

            # Close visualization for the episode
            if self.visualize and episode % self.visualize_interval == 0:
                plt.ioff()
                plt.close()

            # Optional: print episode summary
            # print(f'Episode {episode} completed in {steps} steps with total reward {episode_rewards}')



def choose_action(self, state):
    q_values = self.q_table[state]
    max_q = max(q_values.values())
    probabilities = [np.exp(q / max_q) for q in q_values.values()]
    sum_probabilities = sum(probabilities)
    probabilities = [p / sum_probabilities for p in probabilities]
    action = random.choices(ACTIONS, probabilities)[0]
    return action

    def find_optimal_path(self):
        # After learning, use the Q-table to find the optimal path
        state = self.start
        path = [state]
        steps = 0
        while state != self.end and steps < 1000:
            q_values = self.q_table[state]
            max_value = max(q_values.values())
            max_actions = [action for action, value in q_values.items() if value == max_value]
            action = random.choice(max_actions)

            x, y = state
            dx, dy = ACTION_EFFECTS[action]
            nx, ny = x + dx, y + dy

            # Check for validity
            if (0 <= nx < self.maze.shape[1] and 0 <= ny < self.maze.shape[0]):
                state = (nx, ny)
                path.append(state)
            else:
                break  # Stop if next state is invalid

            steps += 1
        return path

def display_maze(maze, path=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(maze, cmap='binary', interpolation='nearest')
    if path:
        x_coords = [x for x, y in path]
        y_coords = [y for x, y in path]
        plt.plot(x_coords, y_coords, color='blue', linewidth=2)
        plt.scatter(start[0], start[1], color='green', s=100)  # Start
        plt.scatter(end[0], end[1], color='red', s=100)  # End
    plt.xticks([]), plt.yticks([])
    plt.show()

def simulate_agent(agent, delay=0.1):
    state = agent.start
    steps = 0
    path = [state]
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    while state != agent.end and steps < 1000:
        q_values = agent.q_table[state]
        max_value = max(q_values.values())
        max_actions = [action for action, value in q_values.items() if value == max_value]
        action = random.choice(max_actions)
        next_state, _ = agent.take_action(state, action)
        x, y = next_state

        # Display the agent in the maze
        ax.clear()
        ax.imshow(agent.maze, cmap='binary', interpolation='nearest')
        ax.scatter(x, y, color='blue', s=100)
        ax.scatter(agent.start[0], agent.start[1], color='green', s=100)  # Start
        ax.scatter(agent.end[0], agent.end[1], color='red', s=100)  # End
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title('Agent Navigation')
        plt.pause(delay)

        state = next_state
        path.append(state)
        steps += 1

    plt.ioff()
    plt.show()
    return path

# Parameters
width, height = WIDTH, HEIGHT  # Must be odd numbers

# Generate Maze
maze = generate_maze(width, height)

# Define Start and End Points
start = (1, 1)
end = (width - 2, height - 2)

# Create Agent with visualization enabled
agent = Agent(maze, start, end, episodes=30000, alpha=0.7, gamma=0.9, epsilon=0.1, visualize=True, visualize_interval=10)

# Agent learns the maze
print("Agent is learning...")
agent.learn()
print("Learning completed.")

# Find the optimal path after learning
optimal_path = agent.find_optimal_path()

# Display the optimal path
display_maze(maze, optimal_path)

# Optionally, simulate the agent moving through the maze
simulate = input("Do you want to watch the agent navigate the maze? (y/n): ")
if simulate.lower() == 'y':
    print("Simulating agent navigation...")
    simulate_agent(agent)
