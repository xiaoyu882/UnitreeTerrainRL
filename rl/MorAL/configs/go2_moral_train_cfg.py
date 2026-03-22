train_cfg = {
    "runner": {
        "policy_class_name": "ActorCritic",
        "algorithm_class_name": "PPO",
        "num_steps_per_env": 24,
        "save_interval": 200,
        "max_iterations": 2000,
        "export_policy": False,
    },
    "algorithm": {
        "learning_rate": 3e-4,
        "clip_param": 0.2,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "gamma": 0.998,
        "lam": 0.95,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "max_grad_norm": 1.0,
    },
    "policy": {
        "actor_hidden_dims": [256, 256, 256],
        "critic_hidden_dims": [256, 256, 256],
        "activation": "elu",
        "init_noise_std": 1.0,
    },
    "Encoder": {
        # ActorCritic.__init__ 里会用到这些 kwargs（否则 KeyError）
        "priv_info": True,
        "priv_info_dim": 224,
        "HistoryLen": 4,
        "encoder_mlp_units": [180, 128, 12],
    }
}
