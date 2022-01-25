import sacred

############
# Pendulum #
############

pendulum_ingredient = sacred.Ingredient('pendulum')


@pendulum_ingredient.config
def pendulum_config():
    train_dataset_samples = 20000
    test_dataset_samples = 2000
    image_side = 60
    n_stack = 2
    original_frequency = 440.
    sound_velocity = 20.
    sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']


#############
# VAE       #
#############

vae_ingredient = sacred.Ingredient('vae')


@vae_ingredient.config
def vae_config():
    # Training Hyperparameters
    epochs = 500
    batch_size = 128
    learning_rate = 1e-3

    # Capacity parameters
    lambda_image = 1.0
    lambda_sound = 100.0
    beta_image = 1.0
    beta_sound = 1.0
    gamma_image = 10.0
    gamma_sound = 10.0
    beta_top = 1.0
    alpha_fpa = 1.0

    wup_mod_epochs = 0
    wup_top_epochs = 200

    # Model Parameters
    image_latent_dim = 16
    sound_latent_dim = 8
    top_latent_dim = 10

    seed=4


vae_debug_ingredient = sacred.Ingredient('vae_debug')


@vae_debug_ingredient.config
def debug_config():
    artifact_storage_interval = 50


########
# DDPG #
########

ddpg_ingredient = sacred.Ingredient('ddpg')


@ddpg_ingredient.config
def ddpg_config():
    max_frames = 150000

    condition_on_image = False
    condition_on_sound = False
    condition_on_joint = True

    gamma = 0.99
    memory_size = 25000
    max_episode_length = 300
    tau = 0.001
    random_process = {
        'ou_mu': 0.0,
        'ou_theta': 0.15,
        'ou_max_sigma': 0.2,
        'ou_min_sigma': 0.0,
        'ou_decay_period': 100000,
    }

    actor_layers_sizes = [256, 256]
    critic_layers_sizes = [256, 256]

    batch_size = 128
    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3


ddpg_eval_ingredient = sacred.Ingredient('ddpg_eval')


@ddpg_eval_ingredient.config
def ddpg_eval_config():
    # evaluate every x episodes
    eval_frequency = 150
    # number of eval episodes (each time)
    eval_length = 100


ddpg_setup_vae_dep_ingredient = sacred.Ingredient(
    'ddpg_setup_vae_dep')


@ddpg_setup_vae_dep_ingredient.config
def ddpg_setup_vae_dep_config():
    from_file = True
    from_mongodb = False
    file = 'trained_models/muse_last_checkpoint.pth.tar'  # rhang_vae_last_checkpoint.pth.tar

    assert (from_file ^
            from_mongodb) == True, "only one can be true. (xor operation)"
    assert (not from_file) or (
        file is not None), "if from_file, then file cannot be None"





#################
# Eval pipeline #
#################

eval_pipeline_ingredient = sacred.Ingredient('eval_pipeline')


@eval_pipeline_ingredient.config
def eval_pipeline_config():
    eval_episodes = 100
    eval_episode_length = 300

    condition_on_joint = True
    condition_on_image = False
    condition_on_sound = False

    # assert (condition_on_image ^
    #         condition_on_sound), "must be conditioned by a single modality"

    vae_from_file = True
    vae_file = 'trained_models/muse_last_checkpoint.pth.tar'  # rhang_vae_last_checkpoint.pth.tar
    vae_from_mongodb = False

    ddpg_from_file = True
    ddpg_file = 'trained_models/best_ddpg_model.pth.tar'  # ddpg_last_checkpoint.pth.tar
    ddpg_from_mongodb = False

    assert (vae_from_file ^ vae_from_mongodb
            ) == True, "only one can be true. (xor operation)"
    assert (not vae_from_file) or (
        vae_file is not None), "if from_file, then file cannot be None"

    assert (ddpg_from_file ^
            ddpg_from_mongodb) == True, "only one can be true. (xor operation)"
    assert (not ddpg_from_file) or (
        ddpg_file is not None), "if from_file, then file cannot be None"


########
# CUDA #
########

gpu_ingredient = sacred.Ingredient('gpu')


@gpu_ingredient.config
def gpu_config():
    cuda = True
