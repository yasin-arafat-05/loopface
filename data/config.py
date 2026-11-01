# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'loop' : False,
    'haar': False,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 61,
    'decay2': 70,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
    'loop_train_L':(1,6),
    'loop_test_L':4
}

cfg_re50 = {
    'name': 'Resnet50',
    'loop' : False,
    'haar' : False,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256,
    'loop_train_L':(1,6),
    'loop_test_L':4
}

cfg_mnet_loop      = dict(cfg_mnet, loop=True)
cfg_mnet_haar      = dict(cfg_mnet, haar=True)
cfg_mnet_haar_loop = dict(cfg_mnet, haar=True, loop=True)
cfg_mnet_nossh = dict(cfg_mnet, ssh=False, haar=False, loop=False)  
cfg_mnet_loop_nossh = dict(cfg_mnet, ssh=False, haar=False, loop=True)  
cfg_mnet_dloop_nossh = dict(cfg_mnet, ssh=False, haar=False, loop=True, loop_dilations=(1, 2, 4))

# with resnet backbone:
cfg_re50_nossh      = dict(cfg_re50, ssh=False, haar=False, loop=False)
cfg_re50_loop_nossh = dict(cfg_re50, ssh=False, haar=False, loop=True)

# loopcnn:
cfg_mnet_loopcnn_nossh = dict(cfg_mnet, ssh=False, haar=False, loopcnn=True) 

cfg_mnet_loopcnn_ssh = dict(cfg_mnet, ssh=True, haar=False, loopcnn=True)        

# capacity: params incraese, MACs 
cfg_mnet_loopcnn_ssh_untielev  = dict(cfg_mnet_loopcnn_ssh, loopcnn_tie_levels=False)                            # B
cfg_mnet_loopcnn_ssh_untieloop = dict(cfg_mnet_loopcnn_ssh, loopcnn_tie_loops=False)                             # C
cfg_mnet_loopcnn_ssh_untieboth = dict(cfg_mnet_loopcnn_ssh, loopcnn_tie_levels=False, loopcnn_tie_loops=False)   # D

# compute increase: 
cfg_mnet_loopcnn_nossh_k2        = dict(cfg_mnet_loopcnn_ssh, loopcnn_k=2)                                         # E
cfg_mnet_loopcnn_nossh_untielev_k2 = dict(cfg_mnet_loopcnn_ssh, loopcnn_tie_levels=False, loopcnn_k=2)             # F

# memory slots
cfg_mnet_loopcnn_ssh_S1 = dict(cfg_mnet_loopcnn_ssh, loopcnn_slots=1)
cfg_mnet_loopcnn_ssh_S2 = dict(cfg_mnet_loopcnn_ssh, loopcnn_slots=2)
cfg_mnet_loopcnn_ssh_S8 = dict(cfg_mnet_loopcnn_ssh, loopcnn_slots=8)