import sacred

###############
#  Training   #
###############

training_ingredient = sacred.Ingredient('training')


@training_ingredient.config
def training_config():

    # Dataset parameters
    batch_size = 64
    validation_size = 0.1
    eval_samples = 15

    # Training Hyperparameters
    epochs = 100
    learning_rate = 1e-4

    # Capacity parameters
    lambda_mnist = 1.0
    lambda_svhn = 1.0
    lambda_label = 50.0
    beta_mnist = 1.0
    beta_svhn = 1.0
    beta_label = 1.0
    gamma_mnist = 10.0
    gamma_svhn = 10.0
    gamma_label = 10.0
    beta_top = 1.0
    alpha_fpa = 1.0

    wup_mod_epochs = 0
    wup_top_epochs = 0

    # Seed
    seed = 4


##########
# MODEL #
##########

model_ingredient = sacred.Ingredient('model')

@model_ingredient.config
def model_config():

    mnist_latent_dim = 40
    svhn_latent_dim = 40
    label_latent_dim = 4
    top_latent_dim = 16

model_debug_ingredient = sacred.Ingredient('model_debug')

@model_debug_ingredient.config
def debug_config():
    artifact_storage_interval = 10


########
# CUDA #
########

gpu_ingredient = sacred.Ingredient('gpu')


@gpu_ingredient.config
def gpu_config():
    cuda = True



##############
# Evaluation #
##############

evaluation_ingredient = sacred.Ingredient('evaluation')


@evaluation_ingredient.config
def evaluation_config():
    eval_samples = 1000

    file_local = 'trained_models/muse_last_checkpoint.pth.tar'
    file_mongo = 'muse_last_checkpoint.pth.tar'
    from_file = True
    from_mongodb = False

    assert (from_file ^
            from_mongodb) == True, "only one can be true. (xor operation)"
    assert (not from_file) or (
            file_local is not None), "if from_file, then file cannot be None"
