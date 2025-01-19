import numpy as np
import tensorflow as tf
from kaggle_environments import make
import random
import tqdm

# Define parameters
num_actions = 7
num_states = [6, 7]
epsilon = 1.0
epsilon_rate = 0.995

# Create the DQN model
def create_model():
    input_layer = tf.keras.layers.Input(shape=(num_states[0], num_states[1]))
    flatten = tf.keras.layers.Flatten()(input_layer)
    hidden_1 = tf.keras.layers.Dense(50, activation='relu')(flatten)
    hidden_2 = tf.keras.layers.Dense(50, activation='relu')(hidden_1)
    hidden_3 = tf.keras.layers.Dense(50, activation='relu')(hidden_2)
    hidden_4 = tf.keras.layers.Dense(50, activation='relu')(hidden_3)
    output_layer = tf.keras.layers.Dense(num_actions, activation='linear')(hidden_4)
    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Define epsilon decision function
def epsilon_decision(epsilon_value):
    return random.choices(['model', 'random'], weights=[1 - epsilon_value, epsilon_value])[0]

# Define action selection function
def get_action(model, observation, epsilon_value):
    action_decision = epsilon_decision(epsilon_value)
    observation = np.array([observation])
    preds = model.predict(observation)
    weights = tf.nn.softmax(preds).numpy()[0]
    if action_decision == 'model':
        action = np.argmax(weights)
    else:  # Random action
        action = random.randint(0, num_actions - 1)
    return int(action), weights

# Check if the action is valid
def check_valid(obs, action):
    if obs[0, action] != 0:
        valid_actions = set(range(num_actions)) - {action}
        action = random.choice(list(valid_actions))
    return action

# Define reward function
def get_reward(winner, state):
    # Game is not done yet
    if not state:
        return 0.0
    # If the game is finished
    if winner == 1:  # Player 1 (model) wins
        return 50.0
    elif winner == -1:  # Player 2 wins
        return -50.0
    elif winner == 0:  # Draw
        return -50.0
    # Default case (should not happen, but as a fallback)
    return 0.0


# Define experience class
class Experience:
    def __init__(self):
        self.clear()

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []

    def store_experience(self, new_obs, new_act, new_reward):
        self.observations.append(new_obs)
        self.actions.append(new_act)
        self.rewards.append(float(new_reward))  # Ensure reward is a float


# Training step function
def train_step(model, optimizer, observations, actions, rewards):
    with tf.GradientTape() as tape:
        logits = model(observations)
        softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=actions
        )
        loss = tf.reduce_mean(softmax_cross_entropy * rewards)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Main training loop
def train_dqn(episodes=10000):
    env = make("connectx", debug=True)
    optimizer = tf.keras.optimizers.Adam()
    model = create_model()
    exp = Experience()
    global epsilon

    wins = 0
    win_track = []

    for episode in tqdm.tqdm(range(episodes)):
        trainer = env.train([None, 'random'])
        obs = np.array(trainer.reset()['board']).reshape(num_states)
        exp.clear()
        epsilon *= epsilon_rate
        state = False

        while not state:
            action, _ = get_action(model, obs, epsilon)
            action = check_valid(obs, action)
            new_obs, winner, state, _ = trainer.step(action)
            obs = np.array(new_obs['board']).reshape(num_states)
            reward = get_reward(winner, state)
            exp.store_experience(obs, action, reward)

            if state:
                if winner == 1:
                    wins += 1
                win_track.append(wins)
                train_step(
                    model,
                    optimizer,
                    observations=np.array(exp.observations),
                    actions=np.array(exp.actions),
                    rewards=exp.rewards,
                )
                break

    print(f"Training complete. Total wins: {wins}")
    model.save("connect4_model_v1_2.h5")
    model.save_weights("connect4_model_v1_2_weights.h5")
    return model

def evaluate_model(model, num_episodes=100, render=False):
    """
    Evaluates the trained model by running it in the environment for a specified number of episodes.

    Parameters:
        model (tf.keras.Model): The trained DQN model.
        num_episodes (int): Number of episodes to evaluate the model.
        render (bool): Whether to render the environment during evaluation.

    Returns:
        float: Win rate (percentage of games won).
        dict: Detailed evaluation metrics (wins, losses, draws).
    """
    env = make("connectx", debug=True)
    wins, losses, draws = 0, 0, 0

    for episode in range(num_episodes):
        trainer = env.train([None, 'random'])  # Model vs. random opponent
        obs = np.array(trainer.reset()['board']).reshape(num_states)
        done = False

        while not done:
            # Get the action from the model
            action, _ = get_action(model, obs, epsilon_value=0.0)  # epsilon=0 for pure exploitation
            action = check_valid(obs, action)

            # Perform the action in the environment
            next_obs, winner, done, _ = trainer.step(action)
            obs = np.array(next_obs['board']).reshape(num_states)

            if render:
                env.render(mode="ansi")  # Text-based rendering for debugging

        # Track outcomes
        if winner == 1:  # Model wins
            wins += 1
        elif winner == -1:  # Opponent wins
            losses += 1
        else:  # Draw
            draws += 1

    # Compute metrics
    total_games = wins + losses + draws
    win_rate = (wins / total_games) * 100
    evaluation_metrics = {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate
    }

    print(f"Evaluation Results: {evaluation_metrics}")
    return win_rate, evaluation_metrics

def predict_move(model, matrix):
    """
    Predicts the next move for the given Connect 4 board matrix using the trained model.

    Parameters:
        model (tf.keras.Model): The trained DQN model.
        matrix (numpy.ndarray): The current state of the Connect 4 board (6x7 matrix).

    Returns:
        int: The column index (0-6) for the predicted move.
    """
    # Ensure the matrix is properly shaped
    if matrix.shape != (6, 7):
        raise ValueError("Input matrix must have the shape (6, 7).")

    # Add batch dimension to match model input
    input_matrix = np.expand_dims(matrix, axis=0)

    # Get predictions from the model
    predictions = model.predict(input_matrix)

    # Apply softmax to get probabilities
    action_probabilities = tf.nn.softmax(predictions[0]).numpy()

    # Choose the action with the highest probability
    predicted_action = np.argmax(action_probabilities)

    return predicted_action


# Train the model
if __name__ == "__main__":
    """
    trained_model = train_dqn()
    try:
        trained_model.save("connect4_model_v1.h5")
    except Exception as e:
        print(f"Failed to save the model: {e}")"""

    trained_model = tf.keras.models.load_model("connect4_model_v1.h5")

    win_rate, metrics = evaluate_model(trained_model, num_episodes=100)
    print(f"Model Win Rate: {win_rate:.2f}%")




