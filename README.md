# MUSE
Official implementation of the Multimodal Unsupervised Sensing (MUSE) model.

## Citation
```
@article{vasco2021sense,
  title={How to Sense the World: Leveraging Hierarchy in Multimodal Perception for Robust Reinforcement Learning Agents},
  author={Vasco, Miguel and Yin, Hang and Melo, Francisco S and Paiva, Ana},
  journal={arXiv preprint arXiv:2110.03608},
  year={2021}
}
```

------------

## Setup
*Tested on Ubuntu 16.04 LTS, CUDA 10.2*

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

------------

## Experiments

### Multimodal Atari Games

Hyperhot           |  Pendulum
:-------------------------:|:-------------------------:
![](images/hyperhot_game.gif)  |  ![](images/pendulum_game.gif)


#### Training the representation model
To train MUSE:
```
python train_vae.py
```

After training, place the model ``` *_checkpoint.pth.rar``` in the ``` /trained_models``` folder  and rename it  ``` *_last_checkpoint.pth.rar```. 

#### Training the RL algorithm
To train the RL algorithm:
```
python train_rl.py
```

After training, place the model ``` best_*_model.pth.rar``` in the ``` /trained_models``` folder. 


#### Evaluation
To evaluate the performance of the agent:
```
python eval_pipeline.py
```

By default the results are obtained for when the agent
is provided with joint observations. To change the type of observation provided, please select the
appropriate flag in the ``` ingredients.py``` file:

```
@eval_pipeline_ingredient.config
def eval_pipeline_config():
    eval_episodes = 100
    eval_episode_length = 300

    condition_on_joint = True
    condition_on_image = False
    condition_on_sound = False
```

------------

### MNIST/CelebA/MNIST-SVHN evaluation

#### Training
To train MUSE:
```
python train.py
```

After training, place the model ``` *_checkpoint.pth.rar``` in the ``` /trained_models``` folder and rename it  ``` *_last_checkpoint.pth.rar```.  

#### Generation
To generate image samples from labels information (labels):

```python generate.py```
 

#### Evaluation
```
python evaluate_likelihoods.py
```

## FAQ

- To run python scripts without CUDA, add ```with gpu.cuda=False```.
- If ``` ./install_pyenv.sh ``` fails on Ubuntu 18.04 LTS, replace in the file all entries refering to Python ```3.6.4``` with ```3.6.9```.
