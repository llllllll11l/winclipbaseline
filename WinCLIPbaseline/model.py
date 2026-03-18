from typing import List, Tuple

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms
from loguru import logger

from . import CLIPAD
from .adapter import ResidualAdapter
from .ad_prompts import *
valid_backbones = ["ViT-B-16-plus-240"]
valid_pretrained_datasets = ["laion400m_e32"]

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert("RGB")


class WinClipAD(torch.nn.Module):
    """
    WinCLIP-style inference wrapper.

    Keep CLIPAD unchanged for full-image encoding, but build anomaly maps from
    pre-transformer patch embeddings and per-window transformer passes.
    """

    def __init__(
        self,
        out_size_h,
        out_size_w,
        device,
        backbone,
        pretrained_dataset,
        scales,
        precision="fp32",
        **kwargs,
    ):
        super(WinClipAD, self).__init__()

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.device = device
        self.precision = "fp16" if device == "cuda" else "fp32"
        self.scales = tuple(int(scale) for scale in scales)
        self.use_adapter = bool(kwargs.get("use_adapter", False))
        self.adapter_hidden_dim = int(kwargs.get("adapter_hidden_dim", 256))
        self.adapter_dropout = float(kwargs.get("adapter_dropout", 0.0))
        self.adapter_residual_scale = float(kwargs.get("adapter_residual_scale", 1.0))
        self.adapter_checkpoint = kwargs.get("adapter_checkpoint", "")

        self.get_model(backbone, pretrained_dataset, self.scales)
        self.phrase_form = "{}"

        # text feature gallery building version
        # V1: tokens->mean->normalize->
        # V2: tokens->normalize->mean->
        self.version = "V2"

        self.transform = transforms.Compose(
            [
                transforms.Resize((kwargs["img_resize"], kwargs["img_resize"]), Image.BICUBIC),
                transforms.CenterCrop(kwargs["img_cropsize"]),
                _convert_to_rgb,
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_train, std=std_train),
            ]
        )

        self.gt_transform = transforms.Compose(
            [
                transforms.Resize((kwargs["img_resize"], kwargs["img_resize"]), Image.NEAREST),
                transforms.CenterCrop(kwargs["img_cropsize"]),
                transforms.ToTensor(),
            ]
        )

    def get_model(self, backbone, pretrained_dataset, scales):
        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(
            model_name=backbone,
            pretrained=pretrained_dataset,
            scales=scales,
            precision=self.precision,
        )
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval().to(self.device)
        model_dtype = next(model.parameters()).dtype

        self.model = model
        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.text_features = None
        self.grid_size = model.visual.grid_size
        self.feature_dim = int(model.visual.output_dim)
        self.adapter = None
        if self.use_adapter:
            self.adapter = ResidualAdapter(
                embed_dim=self.feature_dim,
                hidden_dim=self.adapter_hidden_dim,
                dropout=self.adapter_dropout,
                residual_scale=self.adapter_residual_scale,
            ).to(device=self.device, dtype=model_dtype)
            if self.adapter_checkpoint:
                self.load_adapter_checkpoint(self.adapter_checkpoint)
        self._prepare_window_masks()
        logger.info(f"grid_size: {self.grid_size}")

    def _prepare_window_masks(self):
        grid_h, grid_w = self.grid_size
        num_patches = grid_h * grid_w  # L = Gh * Gw
        self.window_buffer_names: List[Tuple[str, str]] = []

        for scale_idx, scale in enumerate(self.scales):
            if scale <= 0 or scale > min(grid_h, grid_w):
                raise ValueError(
                    f"Invalid scale {scale} for grid_size={self.grid_size}. "
                    "Each scale is interpreted as a sliding window size in patch units."
                )
            window_idxes = []  # [num_windows, scale * scale]
            window_masks = []  # [num_windows, L]
            for top in range(grid_h - scale + 1):
                for left in range(grid_w - scale + 1):
                    idxes = []
                    for row in range(top, top + scale):
                        row_start = row * grid_w + left
                        idxes.extend(range(row_start, row_start + scale))
                    masks = torch.zeros(num_patches, dtype=torch.float32)
                    masks[idxes] = 1.0
                    window_idxes.append(idxes)
                    window_masks.append(masks)

            idxes_name = f"_window_idxes_{scale_idx}"
            masks_name = f"_window_masks_{scale_idx}"
            # idx tensor: [num_windows, scale * scale]
            self.register_buffer(idxes_name, torch.tensor(window_idxes, dtype=torch.long), persistent=False)
            # mask tensor: [num_windows, L]
            self.register_buffer(masks_name, torch.stack(window_masks, dim=0), persistent=False)
            self.window_buffer_names.append((idxes_name, masks_name))

    def _iter_window_buffers(self):
        for idxes_name, masks_name in self.window_buffer_names:
            yield getattr(self, idxes_name), getattr(self, masks_name)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        if self.precision == "fp16":
            image = image.half()

        image_outputs = self.model.encode_image(image)

        if isinstance(image_outputs, tuple):
            pooled, tokens = image_outputs
            # pooled: [B, D], tokens: [B, L, D]
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)

            visual = self.model.visual
            if hasattr(visual, "ln_post") and visual.ln_post is not None:
                tokens = visual.ln_post(tokens)
            if hasattr(visual, "proj") and visual.proj is not None:
                tokens = tokens @ visual.proj

            tokens = tokens / tokens.norm(dim=-1, keepdim=True)
            return pooled, tokens

        image_features = image_outputs / image_outputs.norm(dim=-1, keepdim=True)
        return image_features, None

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        return self.model.encode_text(text)

    def build_text_feature_gallery(self, category: str):
        normal_phrases = []
        abnormal_phrases = []

        for template_prompt in template_level_prompts:
            for normal_prompt in state_level_normal_prompts:
                phrase = template_prompt.format(normal_prompt.format(category))
                normal_phrases.append(phrase)

            for abnormal_prompt in state_level_abnormal_prompts:
                phrase = template_prompt.format(abnormal_prompt.format(category))
                abnormal_phrases.append(phrase)

        normal_phrases = self.tokenizer(normal_phrases).to(self.device)
        abnormal_phrases = self.tokenizer(abnormal_phrases).to(self.device)

        if self.version == "V1":
            normal_text_features = self.encode_text(normal_phrases)
            abnormal_text_features = self.encode_text(abnormal_phrases)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_phrases.size(0)):
                normal_text_feature = self.encode_text(normal_phrases[phrase_id].unsqueeze(0))
                normal_text_feature = normal_text_feature / normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0)

            abnormal_text_features = []
            for phrase_id in range(abnormal_phrases.size(0)):
                abnormal_text_feature = self.encode_text(abnormal_phrases[phrase_id].unsqueeze(0))
                abnormal_text_feature = abnormal_text_feature / abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0)
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        self.avr_normal_text_features = avr_normal_text_features
        self.avr_abnormal_text_features = avr_abnormal_text_features
        self.text_features = torch.cat([
            self.avr_normal_text_features,
            self.avr_abnormal_text_features,
        ], dim=0)
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def patch_embed(self, image: torch.Tensor) -> torch.Tensor:
        if self.precision == "fp16":
            image = image.half()
        # image: [B, C, H, W]

        visual = self.model.visual
        if visual.input_patchnorm:
            image = image.reshape(
                image.shape[0],
                image.shape[1],
                visual.grid_size[0],
                visual.patch_size[0],
                visual.grid_size[1],
                visual.patch_size[1],
            )
            # [B, C, Gh, Ph, Gw, Pw]

            image = image.permute(0, 2, 4, 1, 3, 5)
            # [B, Gh, Gw, C, Ph, Pw]

            image = image.reshape(image.shape[0], visual.grid_size[0] * visual.grid_size[1], -1)
            # [B, L, C*Ph*Pw]

            image = visual.patchnorm_pre_ln(image)
            image = visual.conv1(image)
            # [B, L, width]

        else:
            image = visual.conv1(image)
            # [B, width, Gh, Gw]
            image = image.reshape(image.shape[0], image.shape[1], -1)
            # [B, width, L]
            image = image.permute(0, 2, 1)
            # [B, L, width]

        return image

    @torch.no_grad()
    def encode_selected_patches(self, patch_embeddings: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        visual = self.model.visual
        # patch_embeddings: [B, L, width]
        # idxs: [num_windows, window_length]
        tokens = patch_embeddings[:, idxs, :]
        # tokens: [B, num_windows, window_length, width]
        batch_size, num_windows, window_length, width = tokens.shape

        pos_embedding = visual.positional_embedding.to(tokens.dtype)
        patch_pos_embedding = pos_embedding[1:][idxs]
        # patch_pos_embedding: [num_windows, window_length, width]
        cls_pos_embedding = pos_embedding[:1].unsqueeze(0).expand(num_windows, -1, -1)
        # cls_pos_embedding: [num_windows, 1, width]

        cls_tokens = visual.class_embedding.to(tokens.dtype).view(1, 1, 1, width)
        cls_tokens = cls_tokens.expand(batch_size, num_windows, 1, width)
        # cls_tokens: [B, num_windows, 1, width]

        x = torch.cat((cls_tokens, tokens), dim=2)
        # [B, num_windows, window_length + 1, width]
        x = x + torch.cat((cls_pos_embedding, patch_pos_embedding), dim=1).unsqueeze(0)
        x = x.reshape(batch_size * num_windows, window_length + 1, width)
        # [B * num_windows, window_length + 1, width]

        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        # [window_length + 1, B * num_windows, width]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        # [B * num_windows, window_length + 1, width]

        if visual.attn_pool is not None:
            x = visual.attn_pool(x)
            x = visual.ln_post(x)
            pooled, _ = visual._global_pool(x)
        else:
            pooled, _ = visual._global_pool(x)
            pooled = visual.ln_post(pooled)

        if visual.proj is not None:
            pooled = pooled @ visual.proj

        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        # [B, num_windows, D]
        return pooled.reshape(batch_size, num_windows, -1)

    @torch.no_grad()
    def encode_window_embeddings(self, patch_embeddings: torch.Tensor) -> List[torch.Tensor]:
        scale_outputs = [[] for _ in self.scales]

        for image_idx in range(patch_embeddings.shape[0]):
            single_image_patches = patch_embeddings[image_idx : image_idx + 1]
            for scale_idx, (window_idxes, _) in enumerate(self._iter_window_buffers()):
                scale_outputs[scale_idx].append(self.encode_selected_patches(single_image_patches, window_idxes))

        # Each [B, num_windows_at_scale, D]
        return [torch.cat(outputs, dim=0) for outputs in scale_outputs]

    def adapt_window_embeddings(self, window_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.adapter is None:
            return window_embeddings
        return [self.adapter(scale_window_embeddings) for scale_window_embeddings in window_embeddings]

    def calculate_textual_anomaly_score(self, window_features: torch.Tensor):
        # window_features: [B, num_windows, D]
        # text_features: [2, D]
        text_features = self.text_features.to(window_features.dtype)

        logits = 100.0 * window_features @ text_features.T
        # logits/probs: [B, num_windows, 2]
        probs = logits.softmax(dim=-1)
        return probs[:, :, 1]
        """
        normal_sim = window_features @ text_features[0].T
        abnormal_sim = window_features @ text_features[1].T
        score = abnormal_sim - normal_sim
        """

    def calculate_textual_anomaly_map(self, window_embeddings: List[torch.Tensor]):
        batch_size = window_embeddings[0].shape[0]
        num_patches = self.grid_size[0] * self.grid_size[1]
        k = 3 # top k

        patch_score_sum = window_embeddings[0].new_zeros(batch_size, num_patches)  # [B, L]
        patch_cover_count = window_embeddings[0].new_zeros(1, num_patches)  # [1, L]
        weighted_patch_score_sum = window_embeddings[0].new_zeros(batch_size, num_patches)
        weighted_sum = window_embeddings[0].new_zeros(batch_size, num_patches)
        scale_patch_scores = []

        for window_features, (_, window_masks) in zip(window_embeddings, self._iter_window_buffers()):
            window_scores = self.calculate_textual_anomaly_score(window_features)
            # window_scores: [B, num_windows]
            window_masks = window_masks.to(device=window_scores.device, dtype=window_scores.dtype)
            # window_masks: [num_windows, L]
            """
            # mean
            patch_score_sum += window_scores @ window_masks
            patch_cover_count += window_masks.sum(dim=0, keepdim=True)
            # weighted mean: weight = window_score ** 2  worse than mean
            weighted_patch_score_sum += (window_scores **2 @ window_masks)
            weighted_sum += window_scores @ window_masks
            """
            masked_scores = window_scores.unsqueeze(-1) * window_masks.unsqueeze(0)
            # [B, num_windows, L]
            masked_scores = masked_scores.masked_fill(window_masks.unsqueeze(0) == 0, float('-inf'))

            # top-k within this scale
            topk_scores, _ = torch.topk(masked_scores, k=k, dim=1)  # [B, k, L]
            valid_mask = torch.isfinite(topk_scores)
            topk_scores = torch.where(valid_mask, topk_scores, torch.zeros_like(topk_scores))
            valid_count = valid_mask.sum(dim=1).clamp_min(1)

            scale_scores = topk_scores.sum(dim=1) / valid_count  # [B, L]
            scale_patch_scores.append(scale_scores)
        # patch_scores: [B, L]
        """
        mean
        patch_scores = patch_score_sum / patch_cover_count.clamp_min(1.0)
        weighted mean: weight = window_score ** 2
        patch_scores = weighted_patch_score_sum / weighted_sum.clamp_min(1e-6)
        """
        # mean across scales
        patch_scores = torch.stack(scale_patch_scores, dim=0).max(dim=0).values  # [B, L]
        # patch_scores = (patch_scores - patch_scores.min(dim=1, keepdim=True)[0]) / (
        #         patch_scores.max(dim=1, keepdim=True)[0] - patch_scores.min(dim=1, keepdim=True)[0] + 1e-6
        # )

        anomaly_map = patch_scores.reshape(batch_size, self.grid_size[0], self.grid_size[1]).unsqueeze(1)
        # anomaly_map: [B, 1, Gh, Gw]
        anomaly_map = F.avg_pool2d(anomaly_map, kernel_size=3, stride=1, padding=1)
        return anomaly_map

    def build_image_feature_gallery(self, images: torch.Tensor):
        raise NotImplementedError("Few-shot image gallery is not implemented")

    def compute_anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        if self.text_features is None:
            raise RuntimeError("Text feature gallery is empty.")

        self.encode_image(images)  # full-image CLIP path kept intact
        patch_embeddings = self.patch_embed(images)
        # patch_embeddings: [B, L, width]
        window_embeddings = self.encode_window_embeddings(patch_embeddings)
        window_embeddings = self.adapt_window_embeddings(window_embeddings)
        anomaly_map = self.calculate_textual_anomaly_map(window_embeddings)
        anomaly_map = F.interpolate(
            anomaly_map,
            size=(self.out_size_h, self.out_size_w),
            mode="bilinear",
            align_corners=False,
        )
        # anomaly_map: [B, 1, out_size_h, out_size_w]
        return anomaly_map

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        anomaly_map = self.compute_anomaly_map(images)
        am_np = anomaly_map.squeeze(1).cpu().numpy()
        return [am_np[i] for i in range(am_np.shape[0])]

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def get_trainable_parameters(self):
        if self.adapter is None:
            return []
        return self.adapter.parameters()

    def get_adapter_state_dict(self):
        if self.adapter is None:
            raise RuntimeError("Adapter is not enabled.")
        return self.adapter.state_dict()

    def load_adapter_checkpoint(self, checkpoint_path: str):
        if self.adapter is None:
            raise RuntimeError("Adapter is not enabled.")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            if "adapter" in checkpoint:
                checkpoint = checkpoint["adapter"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
        self.adapter.load_state_dict(checkpoint, strict=True)
        logger.info(f"Loaded adapter checkpoint from {checkpoint_path}")

    def train_mode(self):
        super().train()
        if self.adapter is None:
            self.model.train()
        else:
            self.model.eval()
            self.adapter.train()

    def eval_mode(self):
        super().eval()
        self.model.eval()
        if self.adapter is not None:
            self.adapter.eval()
