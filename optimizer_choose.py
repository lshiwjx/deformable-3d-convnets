from __future__ import print_function, division

import torch
from util import TimerBlock


def optimizer_choose(model, args):
    with TimerBlock('Optimizer choose') as block:
        params = []
        # tmp = list(model.named_parameters())
        for key, value in model.named_parameters():
            if value.requires_grad:
                if key[8:16] == 'conv_off' or key[8:16] == 'conv_off':
                    # if key[17:24]=='weight':
                    ratio = args.lrr
                    params += [{'params': [value], 'lr': ratio * args.lr,
                                'weight_decay': args.wd * args.wdr}]
                    block.log('lr for {}: {}*{}, wd: {}*{}'.format(key, args.lr, ratio, args.wd,
                                                                   args.wdr))
                elif key[0:2] == 'fc' and args.class_num != args.pre_class_num:
                    ratio = 1
                    params += [{'params': [value], 'lr': ratio * args.lr,
                                'weight_decay': args.wd}]
                    block.log('lr for {}: {}*{}, wd: {}*{}'.format(key, args.lr, ratio, args.wd, 1))
                else:
                    params += [{'params': [value], 'lr': args.lr,
                                'weight_decay': args.wd}]

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params)
            block.log('Using Adam optimizer')
        elif args.optimizer == 'sgd':
            momentum = 0.9
            optimizer = torch.optim.SGD(params, momentum=momentum)
            block.log('Using SGD with momentum ' + str(momentum))
        elif args.optimizer == 'sgd_nev':
            momentum = 0.9
            optimizer = torch.optim.SGD(params, momentum=momentum, nesterov=True)
            block.log('Using SGD with momentum ' + str(momentum) + 'and nesterov')
        else:
            momentum = 0.9
            optimizer = torch.optim.SGD(params, momentum=momentum)
            block.log('Using SGD with momentum ' + str(momentum))
        return optimizer
