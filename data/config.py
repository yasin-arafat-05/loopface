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
    'decay1': 190,
    'decay2': 220,
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

