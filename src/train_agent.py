from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import XRLGuardEnv


FEATURE_PATH = "data/processed/X_train.csv"
LABEL_PATH = "data/processed/category_train.csv"
MODEL_PATH = "models/xrl_guard_ppo"


print("=" * 60)
print("XRL-GUARD PPO TRAINING")
print("=" * 60)

print("\nLoading environment...")

base_env = XRLGuardEnv(
    FEATURE_PATH,
    LABEL_PATH
)

print("Environment loaded.")
print("Training records :", len(base_env.X))
print("Features         :", base_env.n_features)
print("Actions          :", base_env.action_space.n)

print("\nCreating PPO agent...")

env = DummyVecEnv([
    lambda: XRLGuardEnv(
        FEATURE_PATH,
        LABEL_PATH
    )
])

model = PPO(
    "MlpPolicy",
    env,

    # Learning rate
    learning_rate=0.00005,

    # PPO rollout
    n_steps=4096,

    # Mini-batch
    batch_size=256,

    # PPO optimization epochs
    n_epochs=10,

    # Discounting
    gamma=0.99,
    gae_lambda=0.95,

    # PPO clipping
    clip_range=0.2,

    # Moderate exploration
    ent_coef=0.01,

    # Value function coefficient
    vf_coef=0.5,

    # Gradient clipping
    max_grad_norm=0.5,

    policy_kwargs=dict(
        net_arch=dict(
            pi=[128, 128],
            vf=[128, 128]
        )
    ),

    verbose=1,

    # Reproducibility
    seed=42
)

print("\nStarting PPO training...")
print("Total timesteps : 500000")
print("This may take several minutes...\n")

model.learn(
    total_timesteps=500000
)

model.save(MODEL_PATH)

print("\nTraining completed.")

print("Model saved to:")
print(MODEL_PATH)

print("=" * 60)
print("XRL-GUARD PPO TRAINING COMPLETED")
print("=" * 60)