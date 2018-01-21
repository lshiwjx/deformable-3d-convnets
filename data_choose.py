from __future__ import print_function, division

from torch.utils.data import DataLoader
from util import TimerBlock

import dataset


def data_choose(args):
    with TimerBlock("Initializing dataset") as block:
        if args.data == 'ego':
            workers = 4
            data_set_train = dataset.EGOImageFolder('train', args.data_root)
            data_set_val = dataset.EGOImageFolder('val', args.data_root)
        elif args.data == 'jester':
            workers = 4
            data_set_train = dataset.JesterImageFolder('train', args.data_root, args.csv_root)
            data_set_val = dataset.JesterImageFolder('val', args.data_root, args.csv_root)
        elif args.data == 'some':
            workers = 7
            data_set_train = dataset.SomeImageFolder('train', args.data_root, args.csv_root)
            data_set_val = dataset.SomeImageFolder('val', args.data_root, args.csv_root)
        else:
            raise (RuntimeError('No data loader'))
        data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                                     num_workers=workers, drop_last=False, pin_memory=True)
        data_loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=workers, drop_last=True, pin_memory=True)

        block.log('Data load finished: ' + args.data)
        return data_loader_train, data_loader_val
