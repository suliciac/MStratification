"""
MS patients stratification based on EDSS score at cross-sectional timepoint

"""
import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional
import numpy as np
from tqdm import tqdm
import sys
import pandas as pd
from utils import read_yaml, AttrDict, EarlyStopping, accuracy, metrics_, confusion_matrix, unpack_preds, parseargs, \
    load_model
from network import LightClassifier, EncoderForLight, LightClassification
from dataset import dataloader_mult_strat, dataloader_mult_strat_reg, dataloader_mult_inf
import os


def validate(args, model, loader, criteria):
    model.eval()
    val_loss, val_acc = [], []
    with torch.no_grad():
        for batch, (x, tar) in enumerate(tqdm(loader)):
            x = x.type(torch.FloatTensor).to(args.device)
            tar = tar.type(torch.LongTensor).to(args.device)

            # eval
            out = model(x)
            if len(out.shape) < 2:
                out = torch.unsqueeze(out, dim=0)
            loss_val = criteria(out, tar)
            loss_div = loss_val/args.bs_mul
            val_loss.append(loss_div.item())
            out_sqz = torch.argmax(torch.nn.functional.softmax(out, 1), 1)
            val_acc.append(accuracy(out_sqz, tar).item())

    return val_loss, val_acc


def training(args, model, loader, optim, criteria):
    model.train()
    losses, accs = [], []
    optim.zero_grad()
    for batch, (x, tar) in enumerate(tqdm(loader)):
        x = x.type(torch.FloatTensor)
        tar = tar.type(torch.LongTensor)
        x = x.to(args.device)
        tar = tar.to(args.device)

        # forward
        out = model(x)
        if len(out.shape) < 2:
            out = torch.unsqueeze(out, dim=0)
        # backward
        loss = criteria(out, tar)
        loss_div = loss / args.bs_mul
        loss_div.backward()
        out_sqz = torch.argmax(torch.nn.functional.softmax(out, 1), 1)
        accs.append(accuracy(out_sqz, tar).item())
        losses.append(loss_div.item())
        if ((batch % args.bs_mul == 0) and (batch != 0)) or (batch == len(loader)):
            optim.step()
            optim.zero_grad()
    return losses, accs


def testing(args, model, loader):
    act = nn.Softmax()
    predictions, classes = [], []
    with torch.no_grad():
        for bb, (x, cls) in enumerate(loader):
            x = x.type(torch.FloatTensor).to(args.device)
            cls = cls.type(torch.LongTensor)

            # predict
            out = model(x)
            out = act(out)
            predictions.append(out.detach().cpu().numpy())
            classes.append(cls.detach().numpy())

    # metrics
    pred_label = np.array([np.argmax(pred) for pred in predictions])
    class_list = np.array([i[0] for i in classes])
    acc = metrics_(pred_label, class_list)

    return predictions, pred_label, class_list, acc


if __name__ == '__main__':
    # set general options
    pars = parseargs()
    cfg = read_yaml(pars.cfg_file)
    opts = AttrDict(cfg)
    opts.ps = tuple(opts.ps)
    opts.center = tuple(opts.center)
    opts.weights = tuple(opts.weights)
    print('\n ----------------- %s -----------------' % opts.base_out)
    if opts.external:
        test_set = pd.read_csv(opts.external)
    else:
        for f in range(opts.n_folds):
            fold = 'fold_%i' % f
            globals()[fold] = pd.read_csv('%s_%i.csv' % (opts.fold_base, f))

    # initialize vars for results
    probs, preds, classs, accs = [], [], [], []
    subs, sess = [], []

    for exp in range(opts.n_folds):
        print('\n ------------- Starting experiment %i ... -------------' % exp)
        opts.o = '%s_%i' % (opts.base_out, exp)
        # Reproducibility and determinism
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if not opts.external:
            # Set fold to use
            if exp == (opts.n_folds - 1):
                vv = 0
            else:
                vv = exp + 1
            test_set = locals()['fold_%i' % exp]
            val_set = locals()['fold_%i' % vv]
            rest = []
            trtr = [i for i in range(opts.n_folds) if i not in [exp, vv]]
            for i in trtr:
                rest.append(locals()['fold_%i' % i])
            train_set = pd.concat(rest)
            # Print summary of composing data sets
            print('\n -- Test fold: %i ' % exp)
            print('\n -- Validation fold: %i ' % vv)
            print('\n -- Train folds: %s' % trtr)
            print('\n ------------------------------------------------------')

        # Create model
        encoder = EncoderForLight(opts.n_ch, opts.n_filters)
        classifier = LightClassifier(opts.n_cls, k_filters=opts.n_filters*8)
        net = LightClassification(encoder, classifier)

        if os.path.isfile(opts.o):
            path_model = opts.o
            print('\n ----------------- Model already trained ')
            net = load_model(path_model, net)

        else:
            net = net.to(opts.device)
            summary(net, (opts.n_ch,) + opts.ps)
            # Loss function
            weights = torch.tensor(opts.weights).to(opts.device)
            loss_func = nn.CrossEntropyLoss(weight=weights)
            # Optimizer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opts.lr, betas=(0.5, 0.999))
            lambda_sq = lambda epoch: (0.99 ** epoch)
            lr_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_sq)

            # Dataloader
            print('\n ----------------- Building dataloaders ... -------------------')
            if opts.roi:
                train_gen = dataloader_mult_strat_reg(opts, train_set)
                val_gen = dataloader_mult_strat_reg(opts, val_set)
            else:
                train_gen = dataloader_mult_strat(opts, train_set)
                val_gen = dataloader_mult_strat(opts, val_set)

            print('\n ----------------- Start training ... -------------------')
            early_stopping = EarlyStopping(opts.patience, verbose=False, scheduler=lr_decay)
            ttr = tqdm(range(opts.ne))
            for epoch in ttr:
                print('Training epoch %i:' % epoch)
                tr_loss, tr_acc = training(opts, net, train_gen, optimizer, loss_func)
                avg_loss = np.mean(tr_loss)
                avg_acc = np.mean(tr_acc)
                ttr.set_postfix(tr_loss=avg_loss, tr_acc=avg_acc)

                # validate
                vl_loss, vl_acc = validate(opts, net, val_gen, loss_func)
                avg_vl_loss = np.mean(vl_loss)
                avg_vl_acc = np.mean(vl_acc)
                ttr.set_postfix(tr_loss=avg_loss, tr_acc=avg_acc, val_loss=avg_vl_loss, val_acc=avg_vl_acc)

                # test for early stopping
                early_stopping(avg_vl_loss, net, opts.o)
                if early_stopping.early_stop:
                    print('\n Patience Reached - Early Stopping Activated at epoch %s' % epoch)
                    net.eval()
                    break

                elif epoch == opts.ne:
                    print('Finished Training', flush=True)
                    print('Saving the model', flush=True)
                    # save model
                    torch.save(net.state_dict(), opts.o)
                    net.eval()

        print('\n ----------------- Inference ... -------------------')
        # Test dataloader
        if (opts.external and exp == 0) or not opts.external:
            test_gen = dataloader_mult_inf(opts, test_set)
        prob, pred, clas, accy = testing(opts, net, test_gen)
        probs.append(prob)
        preds.append(pred)
        classs.extend(clas)
        accs.append(accy)
        subs.extend(test_set.subject)
        sess.extend(test_set.session)

    print('\n ----------------- Experiment results ... -------------------')
    print('Model %s' % opts.base_out)
    print('avg accuracy: %f' % np.mean(accs))
    print('std: %f' % np.std(accs))
    # confusion matrix
    unpq_preds = unpack_preds(preds)
    conf_matrix = confusion_matrix(classs, unpq_preds)
    print('confusion matrix: \n')
    print(conf_matrix)

    print('\n ----------------- Saving results ... ------------------- ')
    probs_max = [np.max(c) for p in probs for c in p]
    unpq_probs = unpack_preds(probs)
    probs_0 = [pr[0] for pr in unpq_probs]
    probs_1 = [pr[1] for pr in unpq_probs]
    if opts.external:
        opts.base_out = '%s_ext_inf' % opts.base_out

    df_ = pd.DataFrame(
        {'subject': subs, 'test_ses': sess, 'class': classs, 'preds': unpq_preds,
         'probs': probs_max, 'probs_0': probs_0, 'probs_1': probs_1})
    df_.to_csv('preds_%s.csv' % opts.base_out, index=False)

    sys.exit('End of experiment. Have a nice day!')
