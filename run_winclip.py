import argparse
import subprocess
import sys
from pathlib import Path

from datasets import dataset_classes


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def infer_adapter_checkpoint(adapter_root_dir: str, dataset: str, class_name: str, k_shot: int, shared: bool) -> str:
    exp_name = f"{dataset}-k-{k_shot}"
    checkpoint_name = f"{dataset}_shared_adapter_best.pt" if shared else f"{class_name}_adapter_best.pt"
    checkpoint = (
        Path(adapter_root_dir)
        / exp_name
        / "logger"
        / ("all" if shared else class_name)
        / "adapter_ckpts"
        / checkpoint_name
    )
    return str(checkpoint)


def build_eval_command(args, dataset: str, class_name: str):
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
        str(args.batch_size),
        "--vis",
        str(args.vis),
        "--root-dir",
        args.root_dir,
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
    ]

    if args.scales:
        command.extend(["--scales", *[str(scale) for scale in args.scales]])

    if args.use_adapter:
        adapter_checkpoint = args.adapter_checkpoint or infer_adapter_checkpoint(
            adapter_root_dir=args.adapter_root_dir,
            dataset=dataset,
            class_name=class_name,
            k_shot=args.k_shot,
            shared=args.shared_adapter,
        )
        command.extend(
            [
                "--use-adapter",
                "true",
                "--adapter-hidden-dim",
                str(args.adapter_hidden_dim),
                "--adapter-dropout",
                str(args.adapter_dropout),
                "--adapter-residual-scale",
                str(args.adapter_residual_scale),
                "--adapter-checkpoint",
                adapter_checkpoint,
            ]
        )

    return command


def get_args():
    parser = argparse.ArgumentParser(description="Run WinCLIP evaluation across all classes in one or more datasets")
    parser.add_argument("--datasets", nargs="+", default=["mvtec"], choices=["mvtec", "visa"])
    parser.add_argument("--img-resize", type=int, default=240)
    parser.add_argument("--img-cropsize", type=int, default=240)
    parser.add_argument("--resolution", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--vis", type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result_winclip")
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
    parser.add_argument("--use-adapter", type=str2bool, default=False)
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--adapter-residual-scale", type=float, default=1.0)
    parser.add_argument("--adapter-checkpoint", type=str, default="")
    parser.add_argument("--adapter-root-dir", type=str, default="./result_adapter")
    parser.add_argument("--shared-adapter", type=str2bool, default=True)
    return parser.parse_args()


def main():
    args = get_args()

    for dataset in args.datasets:
        for class_name in dataset_classes[dataset]:
            command = build_eval_command(args, dataset, class_name)
            print(" ".join(command))
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
