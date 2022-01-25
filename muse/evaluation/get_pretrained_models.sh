## Game

### Hyperhot
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1T5BH_41zgnCASBIu1kKprXn1TrQG7QZB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1T5BH_41zgnCASBIu1kKprXn1TrQG7QZB" -O ./atari/hyperhot/trained_models/muse_last_checkpoint.pth.tar && rm -rf /tmp/cookies.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TZHK9fVWjTDIAG8eDqbreqMjQCs-jmp9' -O ./atari/hyperhot/trained_models/best_dqn_model.pth.tar


### Pendulum
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BxLD6vY7fjRyTmdZvpPm2JW2m6RjRYeu' -O ./atari/pendulum/trained_models/muse_last_checkpoint.pth.tar
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=11Uyb4y2gES00GLvZPFgB2XAosRJl_OC6' -O ./atari/pendulum/trained_models/best_ddpg_model.pth.tar


## Standard
#### MNIST
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=10HhHdAT_QJxExhLdPULyQWdpcEXZL3-G' -O ./standard/mnist/trained_models/muse_last_checkpoint.pth.tar


#### CelebA
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lYVkSMQ0twlfv6Cs97_oEBoGT_3fo2UG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lYVkSMQ0twlfv6Cs97_oEBoGT_3fo2UG" -O ./standard/celeb/trained_models/muse_last_checkpoint.pth.tar && rm -rf /tmp/cookies.txt


#### MNIST-SVHN
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bKbc-BXvttf1s37jbLpHkEqJqRz4nNy6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bKbc-BXvttf1s37jbLpHkEqJqRz4nNy6" -O ./standard/mnist_svhn/trained_models/muse_last_checkpoint.pth.tar && rm -rf /tmp/cookies.txt
