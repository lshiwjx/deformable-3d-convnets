import train_val_model


def train_net(data_loader, model, loss_function, optimizer, global_step, args, writer):
    if args.train == 'classify':
        return train_val_model.train_classifier(data_loader, model, loss_function, optimizer, global_step, args, writer)
    elif args.train == 'classify_multi_crop':
        return train_val_model.train_classifier(data_loader, model, loss_function, optimizer, global_step, args, writer)
    else:
        raise (RuntimeError("train arg is not right"))


def val_net(data_loader, model, loss_function, global_step, args, writer):
    if args.train == 'classify':
        return train_val_model.val_classifier(data_loader, model, loss_function, global_step, args, writer)
    elif args.train == 'classify_multi_crop':
        return train_val_model.val_classifier_multi_crop(data_loader, model, loss_function, global_step, args, writer)
    else:
        raise (RuntimeError("val arg is not right"))
