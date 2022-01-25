# Multimodal Atari
## Hyperhot
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GiB1O2B-VCiDHbPeRGOmka5Drv6xxiz6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GiB1O2B-VCiDHbPeRGOmka5Drv6xxiz6" -O ./multimodal_atari/data/hyperhot_ds_samples40000_stack2_n_enemies4_pacifist_modeFalse_rec[\'LEFT_BOTTOM\'\,\ \'RIGHT_BOTTOM\'\,\ \'LEFT_SHIP\'\,\ \'RIGHT_SHIP\'].pt && rm -rf /tmp/cookies.txt

## Pendulum
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eBDgbYOKx9ZvSSxt3os3nGk2stMIVK5k' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eBDgbYOKx9ZvSSxt3os3nGk2stMIVK5k" -O ./multimodal_atari/data/pendulum_ds_samples2000_stack2_freq440.0_soundvel20.0_rec[\'LEFT_BOTTOM\'\,\ \'RIGHT_BOTTOM\'\,\ \'MIDDLE_TOP\'].pt && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fwqK4fkvvUsyzBoCVsvMuIHODBOiwZPm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fwqK4fkvvUsyzBoCVsvMuIHODBOiwZPm" -O ./multimodal_atari/data/pendulum_ds_samples20000_stack2_freq440.0_soundvel20.0_rec[\'LEFT_BOTTOM\'\,\ \'RIGHT_BOTTOM\'\,\ \'MIDDLE_TOP\'].pt && rm -rf /tmp/cookies.txt


# Standard
# MNIST
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1XeRkaFQrE9GKy0t4nynR2BzSQOJYsFzL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1XeRkaFQrE9GKy0t4nynR2BzSQOJYsFzL" -O ./standard_dataset/data/MNIST.zip && rm -rf /tmp/cookies.txt

# MNIST-SVHN
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dxwhLkD3IGVe90CV7zNPZKx9xXMkYkZ8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dxwhLkD3IGVe90CV7zNPZKx9xXMkYkZ8" -O ./standard_dataset/data/mm_train.pt && rm -rf /tmp/cookies.txt
#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1te7eP17r-Im93PkHBLvxY3kqmmy7LEbQ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1te7eP17r-Im93PkHBLvxY3kqmmy7LEbQ" -O ./standard_dataset/data/mm_test.pt && rm -rf /tmp/cookies.txt

# CELEBA
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YuJ_lUU33tR4mDJBzCWMjdsiWzveXgFi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YuJ_lUU33tR4mDJBzCWMjdsiWzveXgFi" -O ./standard_dataset/data/celeba/img_align_celeba.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1f-rFcNIrHTecH5_MRK78GHECjDV5UC_N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1f-rFcNIrHTecH5_MRK78GHECjDV5UC_N" -O ./standard_dataset/data/celeba/list_attr_celeba.txt && rm -rf /tmp/cookies.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Lfrg4ITvmuEWI539Ic5A-4ah0RgHcv5N' -O ./standard_dataset/data/celeba/identity_CelebA.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1z18MxEmb_raNxhftaNhG5tmKPfr4D6W9' -O ./standard_dataset/data/celeba/list_bbox_celeba.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c4KBODFP8wr3S7RaYte5qILzm5q45flp' -O ./standard_dataset/data/celeba/list_eval_partition.txt
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Yx0eTnoEO5qlkYK2KLkDa-P2XacJeRUf' -O ./standard_dataset/data/celeba/list_landmarks_align_celeba.txt
