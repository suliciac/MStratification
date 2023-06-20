import yaml
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score
import argparse


def parseargs():
    parser = argparse.ArgumentParser(description='Cross-validation stratification experiment')
    parser.add_argument('-cfg_file', required=True, type=str)
    parser.add_argument('-pred_vent', required=False, type=str, help='Predictions ventricles model for ensemble')
    parser.add_argument('-pred_bsc', required=False, type=str, help='Predictions BSC model for ensemble')
    parser.add_argument('-pred_wm', required=False, type=str, help='Predictions WM model for ensemble')
    parser.add_argument('-pred_gm', required=False, type=str, help='Predictions GM model for ensemble')
    parser.add_argument('-pred_sgm', required=False, type=str, help='Predictions Subcortical-GM model for ensemble')

    return parser.parse_args()


def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_model(path_model, modelclass, device='cuda'):
    model = modelclass
    model.load_state_dict(torch.load(path_model), strict=False)
    model.to(device)
    model.eval()

    return model


def _convert_to_scalar(result, was_numpy):
    if was_numpy:
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy().item()
    return result


def _convert_to_torch(output, target):
    was_numpy = False
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)
        was_numpy = True
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
        was_numpy = True
    return output, target, was_numpy


def accuracy(output, target):
    """Accuracy. Output and target must contain integer labels."""
    output, target, was_numpy = _convert_to_torch(output, target)
    result = torch.mean(torch.eq(output, target).float())
    return _convert_to_scalar(result, was_numpy)


class EarlyStopping:
    # Early stops the training if the validation loss doesn't improve after a given patience
    def __init__(self, patience=5, verbose=False, scheduler=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.scheduler = scheduler

    def __call__(self, val_loss, model, pth):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pth)
        elif score < self.best_score:
            self.counter += 1
            if self.counter == 3:
                print('Learning rate decay')
                self.scheduler.step()
            print('Early Stopping Counter: ', self.counter, '/', self.patience)
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, pth)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        # Saves the model when the validation loss decreases
        if self.verbose:
            print('Validation loss decreased: ', self.val_loss_min, ' --> ', val_loss, 'Saving model ...')

        torch.save(model.state_dict(), path)


def metrics_(single_pred, class_list):
    # confusion matrix
    conf_matrix = confusion_matrix(class_list, single_pred)
    print('confusion matrix: \n')
    print(conf_matrix)
    # balanced accuracy
    bal_acc = balanced_accuracy_score(class_list, single_pred)
    print('balanced accuracy: %f' % bal_acc)
    return bal_acc


def unpack_preds(preds):
    unq_pred = [preds[i][j] for i in range(len(preds)) for j in range(len(preds[i]))]
    return unq_pred
