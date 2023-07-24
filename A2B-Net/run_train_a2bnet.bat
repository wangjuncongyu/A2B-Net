python train_a2bnet.py -backbone=A2BNet-T -use_augm=1 -ssl=0 -save_dir=a2bnet-T_augm1_ssl_no -epoch=100 -lr=0.0001
::python train_a2bnet.py -backbone=A2BNet-B -use_augm=1 -ssl=1 -save_dir=a2bnet-B_augm1_ssl_is -epoch=100 -lr=0.0001
::python train_a2bnet.py -backbone=A2BNet-B -use_augm=1 -ssl=2 -save_dir=a2bnet-B_augm1_ssl_mt -epoch=100 -lr=0.0001
::python train_a2bnet.py -backbone=A2BNet-B -use_augm=1 -ssl=3 -save_dir=a2bnet-B_augm1_ssl_fm -epoch=100 -lr=0.0001
::python train_a2bnet.py -backbone=A2BNet-B -use_augm=1 -ssl=4 -save_dir=a2bnet-B_augm1_ssl_cps -epoch=100 -lr=0.0001
::python train_a2bnet.py -backbone=A2BNet-S -use_augm=1 -ssl=0 -save_dir=a2bnet-B_augm1_ssl_no -epoch=100 -lr=0.0001
::python train_a2bnet.py -backbone=A2BNet-T -use_augm=1 -ssl=0 -save_dir=a2bnet-T_augm1_ssl_no -epoch=100 -lr=0.0001
