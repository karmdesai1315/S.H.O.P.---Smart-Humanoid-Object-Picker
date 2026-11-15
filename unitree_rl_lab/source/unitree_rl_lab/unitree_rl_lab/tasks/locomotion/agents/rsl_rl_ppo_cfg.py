"""PPO settings for the S.H.O.P. goal-walking task.

File: source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py

Same network as Unitree's walking policy (512-256-128, elu). Keep it the same in every stage so each
stage can continue from the previous stage's checkpoint (--resume).
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ShopPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stage 1, trained from scratch."""

    num_steps_per_env = 32  # 32 steps x 0.02 s = 0.64 s of experience per robot per iteration
    max_iterations = 10000
    save_interval = 100
    experiment_name = "shop_g1_stage1"  # checkpoints go to logs/rsl_rl/shop_g1_stage1/

    # normalise the inputs (joints, goal, obstacles have very different ranges).
    # The statistics are saved in the checkpoint, so later stages keep them.
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # Unitree's value: lots of exploration at the start (0.5 was too timid)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Unitree's value: keeps exploring instead of settling on "stand still"
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # Unitree's value; "adaptive" lowers it automatically when updates get too big
        schedule="adaptive",
        gamma=0.995,  # looks ~4 s ahead (0.99 = ~2 s), enough to see the goal from 1-3 m away
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )