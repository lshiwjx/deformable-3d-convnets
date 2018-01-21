# from __future__ import print_function, division

import os
os.environ['DISPLAY'] = 'localhost:10.0'

import shutil
import time
import torch
import setproctitle
# from tensorboard_logger import configure, log_value
from tensorboardX import SummaryWriter
from tqdm import tqdm
from data_choose import data_choose
from lr_scheduler_choose import lr_scheduler_choose
from model_choose import model_choose
from optimizer_choose import optimizer_choose
from tra_val_choose import train_net, val_net
from parser_args import parser_args

# params
args = parser_args()
# args = parser_cifar_args()
# args = parser_flownet_args()
torch.cuda.manual_seed(1)
torch.manual_seed(1)
setproctitle.setproctitle(args.model_saved_name)

print(args.model_saved_name)
if args.mode == 'train_val':
    if os.path.isdir(args.model_saved_name) and not args.use_last_model:
        print('log_dir: ', args.model_saved_name, 'already exist')
        answer = input('delete it? y/n:')
        if answer == 'y':
            shutil.rmtree(args.model_saved_name)
            print('Dir removed: ', args.model_saved_name)
        else:
            print('Dir not removed: ', args.model_saved_name)
    train_writer = SummaryWriter(os.path.join(args.model_saved_name, 'train'), 'train')
    val_writer = SummaryWriter(os.path.join(args.model_saved_name, 'val'), 'val')
else:
    train_writer = val_writer = SummaryWriter(os.path.join(args.model_saved_name, 'test'), 'test')

global_step, model = model_choose(args)

data_loader_train, data_loader_val = data_choose(args)

optimizer = optimizer_choose(model, args)

loss_function = torch.nn.CrossEntropyLoss(size_average=True)

lr_scheduler = lr_scheduler_choose(optimizer, args)

process = tqdm(range(args.max_epoch), 'Process: ' + args.model_saved_name)
for epoch in process:
    last_epoch_time = time.clock()
    model.train()  # Set model to training mode
    global_step = train_net(data_loader_train, model, loss_function, optimizer, global_step, args, train_writer)

    model.eval()
    index = val_net(data_loader_val, model, loss_function, global_step, args, val_writer)

    if args.lr_scheduler == 'reduce_by_epoch':
        lr_scheduler.step(epoch)
    else:
        lr_scheduler.step(index)
    lr = optimizer.param_groups[0]['lr']
    if args.mode == 'train_val':
        train_writer.add_scalar('epoch', epoch, global_step)
        train_writer.add_scalar('lr', lr, global_step)
        train_writer.add_scalar('epoch_time', time.clock() - last_epoch_time, global_step)

    process.set_description_str('Process: ' + args.model_saved_name + ' lr: ' + str(lr), refresh=False)

    # save model
    if (epoch + 1) % args.num_epoch_per_save == 0:
        torch.save(model.state_dict(), args.model_saved_name + '-' + str(global_step) + '.state')

torch.save(model.cpu().state_dict(), args.model_saved_name + '-' + str(global_step) + '.state')
print('Final model: ', args.model_saved_name + '-' + str(global_step) + '.state')
