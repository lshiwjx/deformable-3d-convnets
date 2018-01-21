from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from util import TimerBlock


def lr_scheduler_choose(optimizer, args):
    with TimerBlock('lr scheduler choose ') as block:
        if args.lr_scheduler == 'reduce_by_acc':
            lr_patience = 5
            lr_threshold = 0.001
            lr_delay = 1
            block.log('lr scheduler: lr:{} DecayRatio:{} Patience:{} Threshold:{} Before_epoch:{}'
                      .format(args.lr, args.lr_decay_ratio, lr_patience, lr_threshold, lr_delay))
            return ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_decay_ratio,
                                     patience=lr_patience, verbose=True,
                                     threshold=lr_threshold, threshold_mode='abs',
                                     cooldown=lr_delay)
        elif args.lr_scheduler == 'reduce_by_loss':
            lr_patience = 5
            lr_threshold = 0.001
            lr_delay = 1
            block.log('lr scheduler: lr:{} DecayRatio:{} Patience:{} Threshold:{} Before_epoch:{}'
                      .format(args.lr, args.lr_decay_ratio, lr_patience, lr_threshold, lr_delay))
            return ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_decay_ratio,
                                     patience=lr_patience, verbose=True,
                                     threshold=lr_threshold, threshold_mode='abs',
                                     cooldown=lr_delay)
        elif args.lr_scheduler == 'reduce_by_epoch':
            step = [5, 15, 25]
            block.log('lr scheduler: Reduce by epoch, step: ' + str(step))
            return MultiStepLR(optimizer, step)
        else:
            return None
