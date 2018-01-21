from __future__ import print_function, division
from model import resnet3d
import torch
from util import TimerBlock


def model_choose(args):
    with TimerBlock("Initializing model") as block:
        model = resnet3d.ResNetXt1013d(args.class_num, args.mode)

        if args.use_pre_trained_model:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(args.pre_trained_model)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if args.pre_class_num != args.class_num:
                pretrained_dict.pop('fc.weight')
                pretrained_dict.pop('fc.bias')
            block.log('following weight not load: ' + str(set(model_dict) - set(pretrained_dict)))
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            block.log('Pretrained model load finished: ' + args.pre_trained_model)

        if args.only_train_classifier:
            block.log('Only train classifier')
            # tmp = list(model.named_parameters())
            for key, value in model.named_parameters():
                # print(key)
                value.requires_grad = False
                if key[0:2] == 'fc' or key[0:6] == 'layer4':
                    value.requires_grad = True
                    block.log(key + '-require grad')

        global_step = 0
        # The name for model must be **_**-$(step).state
        if args.use_last_model:
            model.load_state_dict(torch.load(args.last_model))
            global_step = int(args.last_model[:-6].split('-')[1])
            block.log('Training continue, last model load finished, step is ' + str(global_step))

        model.cuda()
        block.log('copy model to gpu')
        return global_step, model
