import argparse
from time import time
from typing import Any
from logging_funtions import log_args
from dataset import get_filepaths
import torch
from model_hub import get_model
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


TRAIN_TS = int(time())


def train(
    batch_size: int = None,
    lr: float = None,
    n_steps: int = None,
    model_save_path: str = None,
    scene_size: int = None,
    device: torch.device = None,
    train_loader: torch.utils.data.DataLoader = None,
    model: Any = None,
):
    wandb.config = {
        "learning_rate": lr,
        "epochs": n_steps,
        "batch_size": batch_size,
        "scene_size": scene_size,
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)
    criterion = torch.nn.CrossEntropyLoss()

    for step in range(n_steps):
        with tqdm(train_loader, unit="step") as tepoch:
            for i, train_batch in enumerate(tepoch):
                tepoch.set_description(f"Step {step}")
                optimizer.zero_grad()
                output_scores = model(train_batch)
                target = torch.stack(train_batch['pt_labs'])
                target = torch.squeeze(target, -1).type(torch.LongTensor).to(device)
                pred_error = criterion(output_scores.permute(0, -1, 1), target)
                wandb.log({
                    "pred_error": pred_error,
                })

                pred_error.backward()
                optimizer.step()
                torch.cuda.empty_cache()

        torch_lr_scheduler.step()
        torch.save(
            {
                'epoch': step,
                'model_state': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': pred_error,
            }, model_save_path.format(TRAIN_TS),
        )


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-steps', type=int, default=500)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--model-save-path', type=str, default=f'/mnt/vol0/datasets/plane_extraction_model_states/saved_models/{TRAIN_TS}.pth')
    parser.add_argument('--train-size', type=float, default=0.75)
    parser.add_argument('--device-name', type=str, default='cuda:0')
    parser.add_argument('--scene-size', type=int, default=120000)
    parser.add_argument('--dataset', type=str, choices=['kitti', 'carla', 'both'], required=True)
    parser.add_argument('--model-name', type=str, choices=['dsnet'], required=True)
    parser.add_argument('--model-state-path', type=str, default='/mnt/vol0/datasets/plane_extraction_model_states/selected_models/dsnet_semantic_full_kitti.pth')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--val', default=False, action='store_true')
    args = parser.parse_args()

    log_args(args)

    train_path, val_path = get_filepaths(args.dataset, args.train_size)
    device = torch.device(args.device_name)
    model, train_loader, val_loader = get_model(
        args.model_name,
        train_path,
        val_path,
        args.scene_size,
        args.batch_size,
        args.device_name,
        args.model_state_path,
        args.n_classes,
    )
    if args.train:
        train(
            args.batch_size,
            args.lr,
            args.n_steps,
            args.model_save_path,
            args.scene_size,
            device,
            train_loader,
            model,
        )