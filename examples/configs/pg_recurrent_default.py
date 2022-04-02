import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.critic_hidden_dims = (128, 128)
    config.actor_hidden_dims = (64,)
    config.actor_recurrent_hidden_dim = 64
    config.use_recurrent_policy = True

    config.discount = 0.99
    config.entropy_coef = 1e-3
    config.mix_coef = 0.01

    return config
