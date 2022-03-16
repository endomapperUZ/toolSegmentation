import json
from datetime import datetime
from pathlib import Path
import time 
import copy
import random
import numpy as np

import torch
import tqdm
from collections import OrderedDict

def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def write_event(log, step, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


def check_crop_size(image_height, image_width):
    """Checks if image size divisible by 32.

    Args:
        image_height:
        image_width:

    Returns:
        True if both height and width divisible by 32 and False otherwise.

    """
    return image_height % 32 == 0 and image_width % 32 == 0


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())

def train(args, model, model_name, num_mod,type_mod, criterion, train_loader, valid_loader, validation, init_optimizer, n_epochs=None, fold=None,
          num_classes=None):
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    optimizer = init_optimizer(lr)
    root = Path(args.root)
    model_path = root / 'data/models_ft/{}_binary_20/model_{}.pt'.format(model_name, num_mod)
    new_model_path = root / 'data/models_ft/{}_binary_20/model_{}_{}.pt'.format(model_name, num_mod, type_mod)
    
    if new_model_path.exists():
        state = torch.load(str(new_model_path))
        epoch = state['epoch']
        step = state['step']
        best_jac = state['jac']
        new_state_dict = OrderedDict()
        for k, v in state['model'].items():
          name = k.replace("module.","") # remove module.
          #print(name)
          new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        #model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    elif model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        best_jac = 0
        new_state_dict = OrderedDict()
        for k, v in state['model'].items():
          name = k.replace("module.","") # remove module.
          #print(name)
          new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        #model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_jac = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'jac': best_jac
    }, str(new_model_path)) # .replace('models','models_ft'))

    report_each = 10
    log = root.joinpath('train_{}_{}.log'.format(model_name,num_mod)).open('at', encoding='utf8')
    valid_losses = []
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        jaccard = []
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs = cuda(inputs)

                with torch.no_grad():
                    targets = cuda(targets)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                jaccard += get_jaccard(targets, (outputs > 0).float())
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                #print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
                if i and i % report_each == 0:                
                    write_event(log, step, loss=mean_loss)
            mean_jac = np.mean(jaccard).astype(np.float64)
            metrics = {'loss': mean_loss, 'jaccard_loss': mean_jac}
            #write_event(log, step, loss=mean_loss)
            write_event(log, step, **metrics)
            tq.close()
            #save(epoch + 1)
            valid_metrics = validation(model, criterion, valid_loader, num_classes)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
            valid_jac = valid_metrics['jaccard_loss']
            if valid_jac > best_jac:
              best_jac = valid_jac
              save(epoch + 1)
            else:
                state['epoch'] = epoch + 1              
            print('Best model jaccard : '+ str(best_jac))
        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return
