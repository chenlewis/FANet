import torch
from utils.datasets_csv import MyDataset
from utils.duq import DUQ
import argparse
import json
import os
from tqdm import tqdm

def main(
    batch_size,
    length_scale,
    centroid_size,
    eta,
    num_classes,
    input_dataset_dir,
    output_dir,
):
    model_output_size = 2
    if centroid_size is None:
        centroid_size = model_output_size

    model = DUQ(
        num_classes,
        batch_size,
        centroid_size,
        model_output_size,
        length_scale,
        eta,
    )
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt")))

    test_dataset = MyDataset(os.path.join(input_dataset_dir, 'test.csv'))
    kwargs = {"num_workers": 0, "pin_memory": True}

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )

    with torch.no_grad():
        model.eval()
        y_q_pred_l = []
        y_f_pred_l = []
        for batch in tqdm(test_loader):
            x_q, x_f, y_q, y_f, y = batch
            x_q, x_f, y_q, y_f, y = x_q.cuda(), x_f.cuda(), y_q.cuda(), y_f.cuda(), y.cuda()
            y_q_pred, y_f_pred, z, y_pred, e_c, Lambda = model(x_q, x_f)
            y_q_pred_l.append(y_q_pred.detach().cpu().numpy())
            y_f_pred_l.append(y_f_pred.detach().cpu().numpy())


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


    main(**kwargs)