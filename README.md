# MUSE
Official implementation of the Multimodal Unsupervised Sensing (MUSE) model.

## Setup/Instalation
Tested on Ubuntu 16.04 LTS, CUDA 10.2:

1. Run ``` ./install_pyenv.sh ``` to install the pyenv environment (requires administrative privilige to install pyenv dependencies)
2. Add the following to your  ``` .bashrc ``` file:
 ``` 
export PATH="$HOME/.poetry/bin:$PATH"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
  ```  
2. Activate the pyenv environment ``` pyenv activate muse_devel ``` (or create a ``` .python-version ``` file);
3. Upgrade pip: ``` pip install --upgrade pip```
4. Install the required dependencies ``` poetry install ```.
5. Install the extra dependencies
 ``` 
pip install pygame imageio imageio-ffmpeg pysine pandas
 ``` 

### Troubleshooting:
- If ``` ./install_pyenv.sh ``` fails on Ubuntu 18.04 LTS, replace in the file all entries refering to Python ```3.6.4``` with ```3.6.9```.

## Download Datasets for Evaluation (~6.5 GB)
To download the datasets used in the evaluation, run (might take a while):
```
cd muse_devel/scenarios/
./get_datasets.sh    
```

After running the script, please extract the ``` MNIST.zip ``` file and the ``` celeba/img_align_celeba.zip ``` file, available in the ``` standard_dataset ``` sub-folder.

Note: in case of error loading the datasets during training, please rerun the script again.
## Download Pretrained models for Evaluation (~0.5 GB)
To download a pretrained version of each model in the evaluation, run:
```
cd muse_devel/evaluation/
./get_pretrained_models.sh    
```

## Experiments

### Multimodal Atari Games

Hyperhot           |  Pendulum
:-------------------------:|:-------------------------:
![](images/hyperhot_game.gif)  |  ![](images/pendulum_game.gif)

The pipeline is identical for both the Pendulum and Hyperhot scenarios.

#### Training the representation model (MUSE)
To train MUSE with CUDA:
```
python train_vae.py
```

To train MUSE without CUDA:
```
python train_vae.py with gpu.cuda=False
```
After training, place the model ``` *_checkpoint.pth.rar``` in the ``` /trained_models``` folder  and rename it  ``` *_last_checkpoint.pth.rar```. 

#### Training the RL algorithm (MUSE)
To train the RL algorithm with CUDA:
```
python train_rl.py
```

To train the RL algorithm without CUDA:
```
python train_rl.py with gpu.cuda=False
```
After training, place the model ``` best_*_model.pth.rar``` in the ``` /trained_models``` folder. 


#### Evaluation (MUSE)
By default the results are obtained for when the agent
is provided only with sound observations. To change the type of observation provided, please select the
appropriate flag in the ``` ingredients.py``` file:

```
@eval_pipeline_ingredient.config
def eval_pipeline_config():
    eval_episodes = 100
    eval_episode_length = 300

    condition_on_joint = False
    condition_on_image = False
    condition_on_sound = True
```

With CUDA:
```
python eval_pipeline.py
```

Without CUDA:
```
python eval_pipeline.py with gpu.cuda=False
```


### MNIST/CelebA/MNIST-SVHN evaluation

#### Training
To train MUSE with CUDA:
```
python train.py
```

To train MUSE without CUDA:
```
python train.py with gpu.cuda=False
```
After training, place the model ``` *_checkpoint.pth.rar``` in the ``` /trained_models``` folder and rename it  ``` *_last_checkpoint.pth.rar```.  

#### Generation
To generate image samples from labels information (labels):

CUDA: ```python generate.py```
 
Without CUDA: ```python generate.py with gpu.cuda=False```


#### Evaluation
##### **Standard likelihood** metrics:

*With CUDA*:
```
python evaluate_likelihoods.py
```

*Without CUDA*:
```
python evaluate_likelihoods.py with gpu.cuda=False
```
```