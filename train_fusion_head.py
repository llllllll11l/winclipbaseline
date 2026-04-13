import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from datasets import dataset_classes, get_dataloader_from_args
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


def resolve_training_classes(dataset: str, class_name: str):
    if class_name == "all":
        return list(dataset_classes[dataset])
    return [class_name]


def build_dataloaders_for_classes(phase, classes, kwargs):
    dataloaders = {}
    for class_name in classes:
        class_kwargs = dict(kwargs)
        class_kwargs["class_name"] = class_name
        dataloader, _ = get_dataloader_from_args(phase=phase, **class_kwargs)
        dataloaders[class_name] = dataloader
    return dataloaders


def compute_losses(model, batch, device, bce):
    images, masks, labels = preprocess_batch(model, batch)
    images = images.to(device)
    masks = masks.to(device)
    labels = labels.to(device)

    anomaly_map = model.compute_anomaly_map(images)
    image_scores = anomaly_map.flatten(1).max(dim=1).values.unsqueeze(1)

    seg_loss = bce(anomaly_map.float().clamp(1e-6, 1 - 1e-6), masks.float())
    cls_loss = bce(image_scores.float().clamp(1e-6, 1 - 1e-6), labels.float())
    return seg_loss, cls_loss


def evaluate(model, dataloaders, device):
    model.eval_mode()
    total_loss = 0.0
    total_batches = 0
    bce = torch.nn.BCELoss()

    with torch.no_grad():
        for class_name, dataloader in dataloaders.items():
            model.build_text_feature_gallery(class_name)
            for batch in dataloader:
                seg_loss, cls_loss = compute_losses(model, batch, device, bce)
                total_loss += (seg_loss + cls_loss).item()
                total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Evaluation dataloader is empty.")
    return total_loss / total_batches


def train(model, train_loaders, val_loaders, device, args, ckpt_dir):
    model.freeze_backbone()

    parameters = list(model.get_fusion_parameters())
    if not parameters:
        raise RuntimeError("Fusion head is not enabled.")

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    bce = torch.nn.BCELoss()
    best_val_loss = float("inf")
    checkpoint_stem = f"{args.dataset}_shared" if args.class_name == "all" else args.class_name
    best_path = os.path.join(ckpt_dir, f"{checkpoint_stem}_fusion_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train_mode()
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_cls_loss = 0.0
        num_batches = 0

        for class_name, train_loader in train_loaders.items():
            model.build_text_feature_gallery(class_name)
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                seg_loss, cls_loss = compute_losses(model, batch, device, bce)
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

        if val_loaders is None:
            val_loss = mean_train_loss
        else:
            val_loss = evaluate(model, val_loaders, device)
            logger.info(f"epoch {epoch}/{args.epochs} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "fusion": model.get_fusion_state_dict(),
                    "class_name": args.class_name,
                    "train_classes": list(train_loaders.keys()),
                    "dataset": args.dataset,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "topk": args.fusion_topk,
                    "scales": list(args.scales),
                },
                best_path,
            )
            logger.info(f"saved best fusion checkpoint to {best_path}")

        if args.save_every_epoch:
            epoch_path = os.path.join(ckpt_dir, f"{checkpoint_stem}_fusion_epoch_{epoch}.pt")
            torch.save({"fusion": model.get_fusion_state_dict(), "epoch": epoch}, epoch_path)

    return best_path, best_val_loss


def get_args():
    argument_parser = argparse.ArgumentParser(description="Train a lightweight fusion head on top of WinCLIP patch score candidates")
    argument_parser.add_argument("--dataset", type=str, default="mvtec", choices=["mvtec", "visa"])
    argument_parser.add_argument("--class-name", type=str, default="all")
    argument_parser.add_argument("--img-resize", type=int, default=240)
    argument_parser.add_argument("--img-cropsize", type=int, default=240)
    argument_parser.add_argument("--resolution", type=int, default=400)
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--root-dir", type=str, default="./result_fusion_head")
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
    argument_parser.add_argument("--train-phase", type=str, default="train_eval",
                                 choices=["train", "test", "train_eval", "val_eval"])
    argument_parser.add_argument("--val-phase", type=str, default="val_eval",
                                 choices=["", "train", "test", "train_eval", "val_eval"])
    argument_parser.add_argument("--fusion-hidden-dim", type=int, default=32)
    argument_parser.add_argument("--fusion-dropout", type=float, default=0.1)
    argument_parser.add_argument("--fusion-topk", type=int, default=3)
    argument_parser.add_argument("--fusion-checkpoint", type=str, default="")
    argument_parser.add_argument("--save-every-epoch", type=str2bool, default=False)
    return argument_parser.parse_args()


def main(args):
    kwargs = vars(args)
    logger.info("Fusion head is trained as a standalone module. Defaults target mvtec zero-shot transfer.")

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
    kwargs["use_fusion"] = True

    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)
    logger.info("==========running parameters=============")
    for k, v in kwargs.items():
        logger.info(f"{k}: {v}")
    logger.info("=========================================")

    train_classes = resolve_training_classes(kwargs["dataset"], kwargs["class_name"])
    logger.info(f"training classes: {train_classes}")

    train_loaders = build_dataloaders_for_classes(args.train_phase, train_classes, kwargs)
    val_loaders = None
    if args.val_phase:
        val_loaders = build_dataloaders_for_classes(args.val_phase, train_classes, kwargs)

    model = WinClipAD(**kwargs).to(device)

    ckpt_dir = os.path.join(logger_dir, "fusion_ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_path, best_val_loss = train(model, train_loaders, val_loaders, device, args, ckpt_dir)
    logger.info(f"best checkpoint: {best_path}")
    logger.info(f"best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parsed_args = get_args()
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{parsed_args.gpu_id}"
    main(parsed_args)
