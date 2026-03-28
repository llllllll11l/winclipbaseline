import argparse
import subprocess
import sys
from pathlib import Path

from datasets import dataset_classes


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def infer_adapter_checkpoint(train_root_dir: str, dataset: str, class_name: str, k_shot: int, shared: bool) -> str:
    exp_name = f"{dataset}-k-{k_shot}"
    checkpoint_name = f"{dataset}_shared_adapter_best.pt" if shared else f"{class_name}_adapter_best.pt"
    checkpoint = (
        Path(train_root_dir)
        / exp_name
        / "logger"
        / ("all" if shared else class_name)
        / "adapter_ckpts"
        / checkpoint_name
    )
    return str(checkpoint)


def build_train_command(args, dataset: str, class_name: str):
    command = [
        sys.executable,
        "train_adapter.py",
        "--dataset",
        dataset,
        "--class-name",
        class_name,
        "--img-resize",
        str(args.img_resize),
        "--img-cropsize",
        str(args.img_cropsize),
        "--resolution",
        str(args.resolution),
        "--batch-size",
        str(args.train_batch_size),
        "--root-dir",
        args.train_root_dir,
        "--experiment_index",
        str(args.experiment_index),
        "--gpu-id",
        str(args.gpu_id),
        "--k-shot",
        str(args.k_shot),
        "--backbone",
        args.backbone,
        "--pretrained_dataset",
        args.pretrained_dataset,
        "--use-cpu",
        str(args.use_cpu),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--seg-loss-weight",
        str(args.seg_loss_weight),
        "--cls-loss-weight",
        str(args.cls_loss_weight),
        "--train-phase",
        args.train_phase,
        "--val-phase",
        args.val_phase,
        "--adapter-hidden-dim",
        str(args.adapter_hidden_dim),
        "--adapter-dropout",
        str(args.adapter_dropout),
        "--adapter-residual-scale",
        str(args.adapter_residual_scale),
        "--save-every-epoch",
        str(args.save_every_epoch),
    ]

    if args.scales:
        command.extend(["--scales", *[str(scale) for scale in args.scales]])

    if args.adapter_checkpoint:
        command.extend(["--adapter-checkpoint", args.adapter_checkpoint])

    return command


def build_eval_command(args, dataset: str, class_name: str, checkpoint_path: str):
    command = [
        sys.executable,
        "eval_WinCLIP.py",
        "--dataset",
        dataset,
        "--class-name",
        class_name,
        "--img-resize",
        str(args.img_resize),
        "--img-cropsize",
        str(args.img_cropsize),
        "--resolution",
        str(args.resolution),
        "--batch-size",
        str(args.eval_batch_size),
        "--vis",
        str(args.vis),
        "--root-dir",
        args.eval_root_dir,
        "--load-memory",
        str(args.load_memory),
        "--cal-pro",
        str(args.cal_pro),
        "--experiment_index",
        str(args.experiment_index),
        "--gpu-id",
        str(args.gpu_id),
        "--pure-test",
        str(args.pure_test),
        "--k-shot",
        str(args.k_shot),
        "--backbone",
        args.backbone,
        "--pretrained_dataset",
        args.pretrained_dataset,
        "--use-cpu",
        str(args.use_cpu),
        "--use-adapter",
        "true",
        "--adapter-hidden-dim",
        str(args.adapter_hidden_dim),
        "--adapter-dropout",
        str(args.eval_adapter_dropout),
        "--adapter-residual-scale",
        str(args.adapter_residual_scale),
        "--adapter-checkpoint",
        checkpoint_path,
    ]

    if args.scales:
        command.extend(["--scales", *[str(scale) for scale in args.scales]])

    return command


def get_args():
    parser = argparse.ArgumentParser(description="Train adapter checkpoints and then evaluate them across all classes")
    parser.add_argument("--datasets", nargs="+", default=["mvtec", "visa"], choices=["mvtec", "visa"])
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--train-root-dir", type=str, default="./result_adapter")
    parser.add_argument("--eval-root-dir", type=str, default="./result_adapter_eval")
    parser.add_argument("--vis", type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=True)
    parser.add_argument("--experiment_index", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--pure-test", type=str2bool, default=False)
    parser.add_argument("--k-shot", type=int, default=0)
    parser.add_argument("--scales", nargs="+", type=int, default=(2, 3))
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240", choices=["ViT-B-16-plus-240"])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--use-cpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seg-loss-weight", type=float, default=1.0)
    parser.add_argument("--cls-loss-weight", type=float, default=0.5)
    parser.add_argument("--train-phase", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--val-phase", type=str, default="test", choices=["", "train", "test"])
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--eval-adapter-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-residual-scale", type=float, default=1.0)
    parser.add_argument("--adapter-checkpoint", type=str, default="")
    parser.add_argument("--save-every-epoch", type=str2bool, default=False)
    parser.add_argument("--skip-train", type=str2bool, default=False)
    return parser.parse_args()


def main():
    args = get_args()

    for dataset in args.datasets:
        shared_checkpoint_path = args.adapter_checkpoint or infer_adapter_checkpoint(
            train_root_dir=args.train_root_dir,
            dataset=dataset,
            class_name="all",
            k_shot=args.k_shot,
            shared=True,
        )

        if not args.skip_train:
            train_command = build_train_command(args, dataset, "all")
            print(" ".join(train_command))
            subprocess.run(train_command, check=True)

        for class_name in dataset_classes[dataset]:
            checkpoint_path = args.adapter_checkpoint or shared_checkpoint_path
            eval_command = build_eval_command(args, dataset, class_name, checkpoint_path)
            print(" ".join(eval_command))
            subprocess.run(eval_command, check=True)


if __name__ == "__main__":
    main()
