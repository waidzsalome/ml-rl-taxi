import gymnasium as gym


def make_taxi_env(seed: int | None = None):
    """
    Create a Taxi-v3 environment.

    Args:
        seed (int | None): random seed for reproducibility

    Returns:
        env: Gymnasium environment
    """
    env = gym.make("Taxi-v3")

    if seed is not None:
        # Gymnasium recommends seeding on reset
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env
