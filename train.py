import argparse
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

IMG_RES = 224
IN_CHANNELS = 25
TL = 80
N_TRAJS = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--dev-data", type=str, required=True, help="Path to rasterized data"
    )
    parser.add_argument(
        "--img-res",
        type=int,
        required=False,
        default=IMG_RES,
        help="Input images resolution",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        required=False,
        default=IN_CHANNELS,
        help="Input raster channels",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        required=False,
        default=TL,
        help="Number time step to predict",
    )
    parser.add_argument(
        "--n-traj",
        type=int,
        required=False,
        default=N_TRAJS,
        help="Number of trajectories to predict",
    )
    parser.add_argument(
        "--save", type=str, required=True, help="Path to save model and logs"
    )

    parser.add_argument(
        "--model", type=str, required=False, default="xception71", help="CNN model name"
    )
    parser.add_argument("--lr", type=float, required=False, default=1e-3)
    parser.add_argument("--batch-size", type=int, required=False, default=48)
    parser.add_argument("--n-epochs", type=int, required=False, default=120)

    parser.add_argument("--valid-limit", type=int, required=False, default=24 * 100)
    parser.add_argument(
        "--n-monitor-train",
        type=int,
        required=False,
        default=10,
        help="Validate model each `n-validate` steps",
    )
    parser.add_argument(
        "--n-monitor-validate",
        type=int,
        required=False,
        default=1000,
        help="Validate model each `n-validate` steps",
    )

    args = parser.parse_args()

    return args


def get_model(model_name, in_ch=IN_CHANNELS, timelimit=TL, n_traj=N_TRAJS):
    if model_name.startswith("efficientnet"):
        model = EfficientNet.from_pretrained(
            "efficientnet-b3",
            in_channels=in_ch,
            num_classes=n_traj * 2 * timelimit + n_traj,
        )
    else:
        model = timm.create_model(
            model_name,
            pretrained=True,
            in_chans=in_ch,
            num_classes=n_traj * 2 * timelimit + n_traj,
        )

    return model


def pytorch_neg_multi_log_likelihood_batch(gt, logits, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        logits (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(
        ((gt - logits) * avails) ** 2, dim=-1
    )  # reduce coords and use availability

    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time

    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)

    return torch.mean(error)


class WaymoLoader(Dataset):
    def __init__(self, directory, limit=0, return_vector=False):
        files = os.listdir(directory)
        self.files = [os.path.join(directory, f) for f in files if f.endswith(".npz")]

        if limit > 0:
            self.files = self.files[:limit]
        else:
            self.files = sorted(self.files)

        self.return_vector = return_vector

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        data = np.load(filename, allow_pickle=True)

        raster = data["raster"].astype("float32")
        raster = raster.transpose(2, 1, 0) / 255

        trajectory = data["gt_marginal"]  # .astype("float32")

        valid = data["future_val_marginal"]

        if self.return_vector:
            return raster, trajectory, valid, data["vector_data"]

        return raster, trajectory, valid


def main():
    args = parse_args()

    summary_writer = SummaryWriter(os.path.join(args.save, "logs"))

    train_path = args.train_data
    dev_path = args.dev_data
    PATH_TO_SAVE = args.save
    if not os.path.exists(PATH_TO_SAVE):
        os.mkdir(PATH_TO_SAVE)

    dataset = WaymoLoader(train_path)

    batch_size = args.batch_size
    num_workers = min(16, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dataset = WaymoLoader(dev_path, limit=args.valid_limit)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True,
    )

    model_name = args.model
    time_limit = args.time_limit
    n_traj = args.n_traj
    model = get_model(
        model_name, in_ch=args.in_channels, timelimit=time_limit, n_traj=n_traj
    )
    model.cuda()

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2 * len(dataloader),
        T_mult=1,
        eta_min=max(1e-2 * lr, 1e-6),
        last_epoch=-1,
    )

    start_iter = 0
    best_loss = float("+inf")
    glosses = []

    tr_it = iter(dataloader)
    n_epochs = args.n_epochs
    progress_bar = tqdm(range(start_iter, len(dataloader) * n_epochs))

    saver = lambda name: torch.save(
        {
            "score": best_loss,
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss.item(),
        },
        os.path.join(PATH_TO_SAVE, name),
    )

    for iteration in progress_bar:
        model.train()
        try:
            x, y, valid = next(tr_it)
        except StopIteration:
            tr_it = iter(dataloader)
            x, y, valid = next(tr_it)

        x, y, valid = map(lambda x: x.cuda(), (x, y, valid))

        optimizer.zero_grad()

        outputs = model(x)
        confidences, logits = outputs[:, :n_traj], outputs[:, n_traj:]
        bs = x.shape[0]
        logits = logits.view(bs, n_traj, time_limit, 2)

        loss = pytorch_neg_multi_log_likelihood_batch(y, logits, confidences, valid)
        loss.backward()
        optimizer.step()
        scheduler.step()

        glosses.append(loss.item())
        if (iteration + 1) % args.n_monitor_train == 0:
            progress_bar.set_description(
                f"loss: {loss.item():.3}"
                f" avg: {np.mean(glosses[-100:]):.2}"
                f" {scheduler.get_last_lr()[-1]:.3}"
            )
            summary_writer.add_scalar("train/loss", loss.item(), iteration)
            summary_writer.add_scalar("lr", scheduler.get_last_lr()[-1], iteration)

        if (iteration + 1) % args.n_monitor_validate == 0:
            optimizer.zero_grad()
            model.eval()
            with torch.no_grad():
                val_losses = []
                for x, y, valid in val_dataloader:
                    x, y, valid = map(lambda x: x.cuda(), (x, y, valid))

                    outputs = model(x)
                    confidences, logits = outputs[:, :n_traj], outputs[:, n_traj:]
                    bs = x.shape[0]
                    logits = logits.view(bs, n_traj, time_limit, 2)
                    loss = pytorch_neg_multi_log_likelihood_batch(
                        y, logits, confidences, valid
                    )
                    val_losses.append(loss.item())

                summary_writer.add_scalar("dev/loss", np.mean(val_losses), iteration)

            saver("model_last.pth")

            mean_val_loss = np.mean(val_losses)
            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                saver("model_best.pth")

                model.eval()
                with torch.no_grad():
                    traced_model = torch.jit.trace(
                        model,
                        torch.rand(
                            1, args.in_channels, args.img_res, args.img_res
                        ).cuda(),
                    )

                traced_model.save(os.path.join(PATH_TO_SAVE, "model_best.pt"))
                del traced_model


if __name__ == "__main__":
    main()
