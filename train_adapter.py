import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from datasets import get_dataloader_from_args
from utils.training_utils import get_dir_from_args, setup_seed
from WinCLIPbaseline import WinClipAD


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def preprocess_batch(model: WinClipAD, batch):
    data, mask, label, _, _ = batch
    images = [
        model.transform(Image.fromarray(cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB)))
        for frame in data
    ]
    images = torch.stack(images, dim=0)

    masks = []
    for gt_mask in mask:
        gt_mask = gt_mask.numpy()
        gt_mask = (gt_mask > 0).astype(np.float32)
        gt_mask = cv2.resize(
            gt_mask,
            (model.out_size_w, model.out_size_h),
            interpolation=cv2.INTER_NEAREST,
        )
        masks.append(torch.from_numpy(gt_mask))

    masks = torch.stack(masks, dim=0).unsqueeze(1)
    labels = label.float().view(-1, 1)
    return images, masks, labels


def evaluate(model, dataloader, device):
    model.eval_mode()
    total_loss = 0.0
    total_batches = 0
    bce = torch.nn.BCELoss()

    with torch.no_grad():
        for batch in dataloader:
            images, masks, labels = preprocess_batch(model, batch)
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            anomaly_map = model.compute_anomaly_map(images)
            image_scores = anomaly_map.flatten(1).max(dim=1).values.unsqueeze(1)

            seg_loss = bce(anomaly_map.clamp(1e-6, 1 - 1e-6), masks)
            cls_loss = bce(image_scores.clamp(1e-6, 1 - 1e-6), labels)
            total_loss += (seg_loss + cls_loss).item()
            total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Evaluation dataloader is empty.")
    return total_loss / total_batches


def train(model, train_loader, val_loader, device, args, ckpt_dir):
    model.freeze_backbone()
    model.build_text_feature_gallery(args.class_name)

    parameters = list(model.get_trainable_parameters())
    if not parameters:
        raise RuntimeError("Adapter is not enabled, nothing to train.")

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCELoss()
    best_val_loss = float("inf")
    best_path = os.path.join(ckpt_dir, f"{args.class_name}_adapter_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train_mode()
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_cls_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images, masks, labels = preprocess_batch(model, batch)
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            anomaly_map = model.compute_anomaly_map(images)
            image_scores = anomaly_map.flatten(1).max(dim=1).values.unsqueeze(1)

            seg_loss = bce(anomaly_map.clamp(1e-6, 1 - 1e-6), masks)
            cls_loss = bce(image_scores.clamp(1e-6, 1 - 1e-6), labels)
            loss = args.seg_loss_weight * seg_loss + args.cls_loss_weight * cls_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_cls_loss += cls_loss.item()
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Training dataloader is empty.")

        mean_train_loss = epoch_loss / num_batches
        mean_seg_loss = epoch_seg_loss / num_batches
        mean_cls_loss = epoch_cls_loss / num_batches
        logger.info(
            f"epoch {epoch}/{args.epochs} train_loss={mean_train_loss:.4f} "
            f"seg_loss={mean_seg_loss:.4f} cls_loss={mean_cls_loss:.4f}"
        )

        if val_loader is None:
            val_loss = mean_train_loss
        else:
            val_loss = evaluate(model, val_loader, device)
            logger.info(f"epoch {epoch}/{args.epochs} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "adapter": model.get_adapter_state_dict(),
                    "class_name": args.class_name,
                    "dataset": args.dataset,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_path,
            )
            logger.info(f"saved best adapter checkpoint to {best_path}")

        if args.save_every_epoch:
            epoch_path = os.path.join(ckpt_dir, f"{args.class_name}_adapter_epoch_{epoch}.pt")
            torch.save({"adapter": model.get_adapter_state_dict(), "epoch": epoch}, epoch_path)

    return best_path, best_val_loss


def get_args():
    argument_parser = argparse.ArgumentParser(description="Train a lightweight adapter on top of WinCLIP window features")
    argument_parser.add_argument("--dataset", type=str, default="visa", choices=["mvtec", "visa"])
    argument_parser.add_argument("--class-name", type=str, default="candle")
    argument_parser.add_argument("--img-resize", type=int, default=240)
    argument_parser.add_argument("--img-cropsize", type=int, default=240)
    argument_parser.add_argument("--resolution", type=int, default=400)
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--root-dir", type=str, default="./result_adapter")
    argument_parser.add_argument("--experiment_index", type=int, default=0)
    argument_parser.add_argument("--gpu-id", type=int, default=0)
    argument_parser.add_argument("--k-shot", type=int, default=0)
    argument_parser.add_argument("--scales", nargs="+", type=int, default=(2, 3))
    argument_parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240", choices=["ViT-B-16-plus-240"])
    argument_parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    argument_parser.add_argument("--use-cpu", type=int, default=0)
    argument_parser.add_argument("--epochs", type=int, default=5)
    argument_parser.add_argument("--lr", type=float, default=1e-4)
    argument_parser.add_argument("--weight-decay", type=float, default=1e-4)
    argument_parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    argument_parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    argument_parser.add_argument("--train-phase", type=str, default="test", choices=["train", "test"])
    argument_parser.add_argument("--val-phase", type=str, default="", choices=["", "train", "test"])
    argument_parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    argument_parser.add_argument("--adapter-dropout", type=float, default=0.1)
    argument_parser.add_argument("--adapter-residual-scale", type=float, default=1.0)
    argument_parser.add_argument("--adapter-checkpoint", type=str, default="")
    argument_parser.add_argument("--save-every-epoch", type=str2bool, default=False)
    return argument_parser.parse_args()


def main(args):
    kwargs = vars(args)
    seeds = [111, 333, 999]
    kwargs["seed"] = seeds[kwargs["experiment_index"]]
    setup_seed(kwargs["seed"])

    if kwargs["use_cpu"] == 0:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = "cpu"
    kwargs["device"] = device
    kwargs["out_size_h"] = kwargs["resolution"]
    kwargs["out_size_w"] = kwargs["resolution"]
    kwargs["use_adapter"] = True

    _, _, logger_dir, _, _ = get_dir_from_args(**kwargs)
    ckpt_dir = os.path.join(logger_dir, "adapter_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info("==========running parameters=============")
    for k, v in kwargs.items():
        logger.info(f"{k}: {v}")
    logger.info("=========================================")

    train_loader, _ = get_dataloader_from_args(phase=args.train_phase, **kwargs)
    val_loader = None
    if args.val_phase:
        val_loader, _ = get_dataloader_from_args(phase=args.val_phase, **kwargs)

    model = WinClipAD(**kwargs).to(device)
    if args.adapter_checkpoint:
        model.load_adapter_checkpoint(args.adapter_checkpoint)

    best_path, best_val_loss = train(model, train_loader, val_loader, device, args, ckpt_dir)
    logger.info(f"best checkpoint: {best_path}")
    logger.info(f"best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    args = get_args()
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    main(args)
