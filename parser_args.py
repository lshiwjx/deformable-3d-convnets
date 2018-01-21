import argparse
import os
from util import TimerBlock
import colorama


def parser_args():
    # params
    parser = argparse.ArgumentParser()
    # resnet3d_18 resnet3d_34 resnetxt3d_101
    # flownets
    parser.add_argument('-model', default='resnetxt3d_101')
    # classify_multi_crop classify
    parser.add_argument('-train', default='classify')
    # ego cifar10 char ucf jester jester_multi some
    parser.add_argument('-data', default='some')
    # train_val test train_test
    parser.add_argument('-mode', default='train_val')
    # cross_entropy mse_ce
    parser.add_argument('-loss', default='cross_entropy')
    # reduce_by_acc reduce_by_loss reduce_by_iteration
    parser.add_argument('-lr_scheduler', default='reduce_by_acc')
    # adam sgd sgd_nev
    parser.add_argument('-optimizer', default='sgd_nev')
    # 83 10 249 101 27 174
    parser.add_argument('-class_num', default=174)
    parser.add_argument('-batch_size', default=70)  # 20
    parser.add_argument('-hidden_size', default=1024)
    parser.add_argument('-wd', default=5e-4)  # 5e-4
    parser.add_argument('-wdr', default=1)
    parser.add_argument('-max_epoch', default=60)

    parser.add_argument('-lr', default=0.001)  # 0.001
    parser.add_argument('-lr_decay_ratio', default=0.1)
    parser.add_argument('-lrr', default=1)

    parser.add_argument('-num_epoch_per_save', default=2)
    parser.add_argument('-model_saved_name', default='./runs/some_new')

    parser.add_argument('-use_last_model', default=False)
    parser.add_argument('-last_model',
                        default='./runs/jester_rext101_nopre-23712.state')

    parser.add_argument('-use_pre_trained_model', default=True)
    parser.add_argument('-pre_class_num', default=174)
    # /opt/model/jester_res3d18_sgd_pre_maxacc-9880.state /opt/model/cifar10_base_100_0.001_0.0005_p3-20000.state
    # /opt/model/resnet3d_18_kinetics.state  /opt/model/resnet3d_34_kinetics.state
    # /opt/model/resnetxt3d_101_kinetics.state ./runs/resxt3d101_ucf_fc10_lr0001-3725.state
    # ./runs/ego_rext101-39950.state
    parser.add_argument('-pre_trained_model', default='./runs/some_new-18420.state')
    parser.add_argument('-only_train_classifier', default=False)

    # parser.add_argument('-clip_length', default=32)
    # parser.add_argument('-resize_shape', default=[160, 120])  # 160, 120  171, 128
    # parser.add_argument('-crop_shape', default=[96, 96])  # 128,96  112, 112  96, 96
    # cha[0.486,0.423,0.45] ego[0.45, 0.48, 0.49] ucf[0.39,0.38,0.35] jester[0.45, 0.48, 0.49]
    # parser.add_argument('-mean', default=[0.45, 0.48, 0.49])

    parser.add_argument('-device_id', default=[0, 1, 2, 3, 4, 5, 6])
    os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4,3,2,1'
    # parser.add_argument('-device_id', default=[0, 1])
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'
    # parser.add_argument('-device_id', default=[0])
    # os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    with TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE', action='store_true')
        # 会返回列表
        defaults = vars(parser.parse_args(['--IGNORE']))
        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))
            block.log2file('./log_file/' + args.model_saved_name.split('/')[-1] + '.txt',
                           '{}: {}'.format(argument, value))
    return args
