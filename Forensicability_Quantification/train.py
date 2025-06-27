import argparse
import json
import os

import torch
import torch.nn.functional as F
import torch.utils.data

from ignite.engine import Events, Engine
from ignite.metrics import Average
from ignite.contrib.handlers import ProgressBar

from utils.datasets_csv import MyDataset
from utils.duq import DUQ

e_c_l = [[1.0, 1.0], [1.0, 0.0], [0.0, 0.5]]
Lambda_c = [1., 1., 1.]

def main(
    batch_size,
    length_scale,
    centroid_size,
    learning_rate,
    eta,
    weight_decay,
    num_classes,
    input_dataset_dir,
    output_dir,
):

    train_dataset = MyDataset(os.path.join(input_dataset_dir, 'train.csv'))

    model_output_size = 2
    epochs = 50
    milestones = [12, 25, 37]

    if centroid_size is None:
        centroid_size = model_output_size

    model = DUQ(
        num_classes,
        batch_size,
        centroid_size,
        length_scale,
        eta,
    )
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    def step(engine, batch):
        model.train()

        optimizer.zero_grad()

        x_q, x_f, y_q, y_f, y = batch
        x_q, x_f, y_q, y_f, y = x_q.cuda(), x_f.cuda(), y_q.cuda(), y_f.cuda(), y.cuda()

        x_q.requires_grad_(True)
        x_f.requires_grad_(True)

        y_q_pred, y_f_pred, z, y_pred, e_c, Lambda = model(x_q, x_f)

        y_pred = y_pred.double()

        y = F.one_hot(y, num_classes).double()
        loss = F.binary_cross_entropy(y_pred, y, reduction="mean") + F.mse_loss(y_q_pred, y_q.unsqueeze(1), reduction="mean") + F.mse_loss(y_f_pred, y_f.unsqueeze(1), reduction="mean")
        # loss = F.mse_loss(y_q_pred, y_q.unsqueeze(1), reduction="mean") + F.mse_loss(y_f_pred, y_f.unsqueeze(1), reduction="mean")

        loss.backward()
        optimizer.step()

        x_q.requires_grad_(False)
        x_f.requires_grad_(False)

        with torch.no_grad():
            model.eval()
            model.update_ec(z, y)
        
        global e_c_l
        e_c_l = e_c.cpu().detach().numpy().tolist()
        global Lambda_c
        Lambda_c = Lambda.cpu().detach().numpy().tolist()
        return loss.item()

    trainer = Engine(step)

    metric = Average()
    metric.attach(trainer, "loss")

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    kwargs = {"num_workers": 0, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        loss = metrics["loss"]
        print(f"Train - Epoch: {trainer.state.epoch} Loss: {loss:.4f}")
        if trainer.state.epoch == epochs:
            print("e_1: [{:.4f}, {:.4f}]".format(e_c_l[0][0], e_c_l[0][1]))
            print("e_2: [{:.4f}, {:.4f}]".format(e_c_l[1][0], e_c_l[1][1]))
            print("e_3: [{:.4f}, {:.4f}]".format(e_c_l[2][0], e_c_l[2][1]))
            print("Lambda_c: [{:.4f}, {:.4f}, {:.4f}]".format(abs(Lambda_c[0]), abs(Lambda_c[1]), abs(Lambda_c[2])))

        scheduler.step()

    trainer.run(train_loader, max_epochs=epochs)
    torch.save(model.state_dict(), f"{output_dir}/model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training (default: 128)",
    )

    parser.add_argument(
        "--centroid_size",
        type=int,
        default=None,
        help="Size to use for centroids (default: same as model output)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate",
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.999,
        help="Decay factor for exponential average",
    )

    parser.add_argument(
        "--length_scale",
        type=float,
        default=0.1,
        help="Length scale of RBF kernel",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Set the number of central categories for the classification",
    )

    parser.add_argument(
        "--input_dataset_dir", type=str, default="data", help="set output folder"
    )

    parser.add_argument(
        "--output_dir", type=str, default="models", help="set input folder"
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

    os.makedirs(args.output_dir, exist_ok=True)

    main(**kwargs)