#python3 train_VAE.py --x ball --m bvae --g 0.0 --b 1 --z 4 --e 10 --l 0.0
#python3 train_DS.py --x ball --m bvae --g 0.0 --b 1 --z 4 --e 10 --l 0.0 --subset 1 --seed 0 --ds_epochs 10 --ds_task 'vel'

#python3 train_VAE.py --x pong --m bvae --g 0.0 --b 1 --z 8 --e 10 --l 0.0
#python3 train_DS.py --x pong --m bvae --g 0.0 --b 1 --z 8 --e 10 --l 0.0 --subset 1 --seed 0 --ds_epochs 10 --ds_task 'actions'

python3 train_VAE.py --x dml --m bvae --g 0.0 --b 1 --z 16 --e 10 --l 0.0
python3 train_DS.py --x dml --m bvae --g 0.0 --b 1 --z 16 --e 10 --l 0.0 --subset 1 --seed 0 --ds_epochs 10 --ds_task 'labels'
