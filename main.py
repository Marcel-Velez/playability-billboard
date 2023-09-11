# Neural Networks
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

from data import retrieve
from utils import retrieve_model


def boundary_loss(input_torch, target, class_total=False):
    input_torch = nn.Softmax()(input_torch)
    if class_total:
        for index in range(input_torch.size(1)):
            if index == 0:
                losses = input_torch[:,index] * abs(target - index)
            else:
                losses += input_torch[:,index] * abs(target - index)

        return torch.mean(losses)
    else:
        return torch.mean(input_torch[:,0] * target + input_torch[:,1] * abs(target-1) + input_torch[:,2] * abs(target-2) + input_torch[:,3] * abs(target-3))


def custom_test(all_models, all_data):
    for target_index in range(7):
        cur_data = all_data[target_index].test_dataloader()
        model = all_models[target_index]
        model.freeze()
        cur_batch_scores = []
        true_batch_sores = []
        for batch in cur_data:
            x, y, x_len = batch
            pred = model.forward(x, x_len).cpu()

            bound_loss = boundary_loss(pred, y)
            pred_class = pred.argmax(1)

            multiplier = [3, 2, 3, 2, 1, 1, 2]
            pred_class *= multiplier[target_index]
            y_true = y * multiplier[target_index]

            cur_batch_scores.append(pred_class)
            true_batch_sores.append(y_true)

        if target_index == 0:
            total_loss_score = bound_loss
            total_pred_score = pred_class
            total_true_score = y_true

        else:
            total_loss_score += bound_loss
            total_pred_score += pred_class
            total_true_score += y_true

    accuracy_score = (total_pred_score == total_true_score).sum()/ total_true_score.size(0)
    accuracy_loss = boundary_loss( F.one_hot(torch.tensor(total_pred_score).to(torch.int64), 42).float(), total_true_score, True)
    return total_loss_score, accuracy_loss, accuracy_score


ALL_TARGETS = ['CFP', 'CFD', 'UC', 'RHC', 'CPT', 'BD', 'R', 'weighted_total']


def main(config):
    seed_everything(1)

    data = retrieve('billboard_salami')

    # gather the data, either for all categories or for a single category
    if config['target'] == "all":
        all_data = []
        for p in ALL_TARGETS:
            dm_cat = data(
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                encoding=config['chord_encoding'],
                target=f"{p}",
                target_data_path=config['file_path'],
            )
            dm_cat.custom_setup(config['k_fold'])
            all_data.append(dm_cat)
    else:
        dm_cat = data(
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            encoding=config['chord_encoding'],
            target=config['target'],
            target_data_path=config['file_path'],
        )
        dm_cat.custom_setup(config['k_fold'])

    # initiate the models, either for all categories or for a single category
    if config['target'] == 'all':
        lot_models = []
        for cat_ind in range(len(ALL_TARGETS)):
            model = retrieve_model(
                config['model'],
                input_size=dm_cat.encode_dict_size,
                learning_rate=config['learning_rate'],
                cat=cat_ind,
                symb_enc=config['chord_encoding'],
                device='cuda' if config['accelerator'] == 'gpu' else 'cpu',
                train=True
            )
            lot_models.append(model)
    else:
        model = retrieve_model(
            config['model'],
            input_size=dm_cat.encode_dict_size,
            learning_rate=config['learning_rate'],
            cat=ALL_TARGETS.index(config['target']),
            symb_enc=config['chord_encoding'],
            device='cuda' if config['accelerator'] == 'gpu' else 'cpu',
            train=True
        )

    csv_logger = WandbLogger(offline=True, project="snellius-models-newlstm")
    csv_logger.experiment.config["k-fold-index"] = config['k_fold']

    # train the models, either for all categories or for a single category
    if config['target'] == "all":
        for p in range(len(ALL_TARGETS)):
            cur_model, cur_data = lot_models[p], all_data[p]
            trainer = Trainer(
                max_epochs=config['max_epochs'],
                logger=csv_logger,
                log_every_n_steps=1,
                accelerator=config['accelerator'],
                devices=1,
            )
            trainer.fit(cur_model, cur_data)
            cur_test = trainer.test(cur_model, datamodule=cur_data)
            print(f"target: {p} test: {cur_test}")

        for i, m in enumerate(lot_models):
            target_index = i+1
            torch.save(m.state_dict(), f"./trained_models/new_{config['model']}_target_{target_index}_fold_{config['k_fold']}_{config['max_epochs']}_{config['chord_encoding']}.pth")

        total_acc = custom_test(lot_models, all_data)
        print("total accuracy: ", total_acc)

    else:
        trainer = Trainer(
            max_epochs=config['max_epochs'],
            logger=csv_logger,
            log_every_n_steps=1,
            accelerator=config['accelerator'],
            devices=1,
            gradient_clip_val=0.5,
        )
        trainer.fit(model, dm_cat)
        cur_test = trainer.test(model, datamodule=dm_cat)
        print(cur_test)
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--chord_encoding', type=str, default='char')
    parser.add_argument('--target', type=str, default='weighted_total')
    parser.add_argument('--k_fold', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.002)
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--dropout',type=float, default=0.5)
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--accelerator', type=str, default='cpu')
    parser.add_argument('--file_path', type=str, default='./data/Annotations.csv')

    args = vars(parser.parse_args())
    print(f'starting run with {args}')

    wandb.init(config=args, project="playability-billboard-chordify-newlstm", name=f"{args['model']}_{args['chord_encoding']}_{args['target']}_{args['k_fold']}_{args['max_epochs']}")

    main(args)
