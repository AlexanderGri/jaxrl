import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4

    config.critic_hidden_dims = (128, 128)
    config.actor_hidden_dims = (128, 128)

    config.discount = 0.99
    config.entropy_coef = 1e-4

    return config
