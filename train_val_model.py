import torch
from torch.autograd import Variable
import torch.nn as nn
from tqdm import tqdm
from util import IteratorTimer
import torchvision


def train_classifier(data_loader, model, loss_function, optimizer, global_step, args, writer):
    process = tqdm(IteratorTimer(data_loader), desc='Train: ')
    for inputs, labels in process:
        inputs, labels = Variable(inputs.cuda(async=True)), Variable(labels.cuda(async=True))

        # use DataParallel to realize multi gpu
        net = torch.nn.DataParallel(model, device_ids=args.device_id)
        outputs, layers = net(inputs)
        if args.mode == 'train_test':
            img = inputs.data.cpu().permute(2, 0, 1, 3, 4)
            img = torchvision.utils.make_grid(img[4], normalize=True)
            writer.add_image('img', img, global_step=global_step)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        value, predict_label = torch.max(outputs.data, 1)
        ls = loss.data[0]
        acc = torch.mean((predict_label == labels.data).float())

        process.set_description_str(
            'Train: acc: {:4f}, loss: {:4f}, batch time: {:4f}'.format(acc, ls, process.iterable.last_duration),
            refresh=False)

        # 每个batch记录一次
        if args.mode == 'train_val':
            writer.add_scalar('acc', acc, global_step)
            writer.add_scalar('loss', ls, global_step)
            writer.add_scalar('batch_time', process.iterable.last_duration, global_step)

    process.close()
    return global_step


def val_classifier_multi_crop(data_loader, model, loss_function, global_step, args, writer,
                              log_file='./wrong_list.txt'):
    ls_total = []
    right_num = 0
    total_num = 0
    process = tqdm(IteratorTimer(data_loader), desc='Val: ')
    f = open(log_file, 'w')
    for inputs, label, path in process:
        output = torch.zeros((label.size(0), args.class_num)).type(torch.cuda.FloatTensor)
        label = Variable(label.cuda(async=True), volatile=True)
        ls = 0
        for index, input in enumerate(inputs):
            input = Variable(input.cuda(async=True), volatile=True)
            net = torch.nn.DataParallel(model, device_ids=args.device_id)
            outputs, layers = net(input)
            loss = loss_function(outputs, label)
            ls += loss.data[0]
            output += outputs.data

        _, predict_label = torch.max(output, 1)
        ls /= (index + 1)
        ls_total.append(ls)
        right_num += torch.sum(predict_label == label.data)

        predict = list(predict_label.cpu().numpy())
        true = list(label.data.cpu().numpy())
        for i, x in enumerate(predict):
            if x != true[i]:
                f.write(path[i].split('/')[-2] + ',' + str(x) + ',' + str(true[i]) + '\n')

        total_num += label.data.size(0)

        process.set_description_str(
            'Val: right num: {:4f}, loss: {:4f}, batch time: {:4f}'.format(right_num, ls,
                                                                           process.iterable.last_duration),
            refresh=False)

    f.close()
    acc = right_num / total_num
    ls = sum(ls_total) / len(ls_total)
    # 每个epoch记录一次
    if args.mode == 'train_val' and writer is not None:
        writer.add_scalar('loss', ls, global_step)
        writer.add_scalar('acc', acc, global_step)

    if args.lr_scheduler == 'reduce_by_acc':
        return acc
    elif args.lr_scheduler == 'reduce_by_loss':
        return ls


def val_classifier(data_loader, model, loss_function, global_step, args, writer):
    right_num_total = 0
    total_num = 0
    loss_total = 0
    step = 0
    process = tqdm(IteratorTimer(data_loader), desc='Val: ')
    for inputs, labels in process:
        inputs, labels = Variable(inputs.cuda(async=True), volatile=True), Variable(labels.cuda(async=True),
                                                                                    volatile=True)
        net = torch.nn.DataParallel(model, device_ids=args.device_id)
        outputs, _ = net(inputs)

        # return value and index
        _, predict_label = torch.max(outputs.data, 1)
        loss = loss_function(outputs, labels)

        right_num = torch.sum(predict_label == labels.data)
        batch_num = labels.data.size(0)
        acc = right_num / batch_num
        ls = loss.data[0]

        right_num_total += right_num
        total_num += batch_num
        loss_total += ls
        step += 1

        process.set_description_str(
            'Val: acc: {:4f}, loss: {:4f}, time: {:4f}'.format(acc, ls, process.iterable.last_duration), refresh=False)

    process.close()
    loss = loss_total / step
    accuracy = right_num_total / total_num
    # 每个epoch记录一次
    if args.mode == 'train_val':
        writer.add_scalar('loss', loss, global_step)
        writer.add_scalar('acc', accuracy, global_step)
        writer.add_scalar('batch time', process.iterable.last_duration, global_step)

    if args.lr_scheduler == 'reduce_by_acc':
        return accuracy
    elif args.lr_scheduler == 'reduce_by_loss':
        return loss



