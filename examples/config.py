import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.save_dir = './tmp'
    config.eval_episodes = 10
    config.log_interval = int(1e3)
    config.replay_interval = int(5e4)
    config.eval_interval = int(1e4)
    config.save_interval = int(1e5)
    config.max_steps = int(1e6) + 1
    config.save_replay = True

    config.map_name = '2s3z'
    config.seed = 45
    config.trajectories_per_update = 1

    config.use_recurrent_policy = True
    config.use_meta_rewards = True

    config.penalty_per_step = 1.

    config.learner_kwargs = {
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'critic_hidden_dims': (128, 128),
        'discount': 0.99,
        'entropy_coef': 1e-3
    }

    config.recurrent_policy_kwargs = {
        'actor_hidden_dims': (64,),
        'actor_recurrent_hidden_dim': 64
    }
    config.policy_kwargs = {
        'actor_hidden_dims': (128, 128)
    }
    config.meta_kwargs = {
        'mix_coef': 0.01,
        'use_shared_reward': False
    }

    return config
