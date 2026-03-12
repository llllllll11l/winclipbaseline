import torch
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
from PIL import Image
from torchvision import transforms

valid_backbones = ['ViT-B-16-plus-240']
valid_pretrained_datasets = ['laion400m_e32']

mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert('RGB')


class WinClipAD(torch.nn.Module):
    """
    CLIP baseline variant:
      - keep prompt gallery
      - remove window-based visual tower logic
      - remove visual gallery
      - remove fusion
      - use normal VisionTransformer patch tokens to build anomaly map
    """

    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, scales, precision='fp32', **kwargs):
        super(WinClipAD, self).__init__()

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = 'fp16'  # keep original project setting
        self.device = device

        self.get_model(backbone, pretrained_dataset, scales)
        self.phrase_form = '{}'

        # text feature gallery building version
        # V1: tokens->mean->normalize->
        # V2: tokens->normalize->mean->
        self.version = 'V2'

        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train),
        ])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor(),
        ])

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

        self.model = model
        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.text_features = None
        self.grid_size = model.visual.grid_size
        print('self.grid_size', self.grid_size)

    @torch.no_grad()
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):
        if self.precision == 'fp16':
            image = image.half()

        image_outputs = self.model.encode_image(image)

        if isinstance(image_outputs, tuple):
            pooled, tokens = image_outputs

            # pooled 一般已经在最终空间，先归一化
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)

            # project patch tokens
            visual = self.model.visual
            # 如果有 ln_post，先做 layer norm
            if hasattr(visual, 'ln_post') and visual.ln_post is not None:
                tokens = visual.ln_post(tokens)
            # 如果有 proj，投影到 CLIP embedding space
            if hasattr(visual, 'proj') and visual.proj is not None:
                tokens = tokens @ visual.proj

            # 归一化
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

        if self.version == 'V1':  #always V1
            normal_text_features = self.encode_text(normal_phrases)
            abnormal_text_features = self.encode_text(abnormal_phrases)
        elif self.version == 'V2':
            normal_text_features = []
            for phrase_id in range(normal_phrases.size(0)):
                normal_text_feature = self.encode_text(normal_phrases[phrase_id].unsqueeze(0))
                normal_text_feature = normal_text_feature / normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()

            abnormal_text_features = []
            for phrase_id in range(abnormal_phrases.size(0)):
                abnormal_text_feature = self.encode_text(abnormal_phrases[phrase_id].unsqueeze(0))
                abnormal_text_feature = abnormal_text_feature / abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
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

    def calculate_textual_anomaly_score(self, patch_tokens: torch.Tensor):
        """
        patch_tokens: [B, L, D]
        text_features: [2, D]  -> [normal_proto, abnormal_proto]
        """
        B, L, D = patch_tokens.shape
        expected_L = self.grid_size[0] * self.grid_size[1]
        if L != expected_L:
            raise RuntimeError(
                f'Patch token length mismatch: got {L}, expected {expected_L} from grid_size={self.grid_size}. '
                'Please make sure the visual tower is the normal ViT and output_tokens=True.'
            )

        text_features = self.text_features.to(patch_tokens.dtype)
        logits = 100.0 * patch_tokens @ text_features.T   # [B, L, 2]
        probs = logits.softmax(dim=-1)
        anomaly_score = probs[:, :, 1]                    # [B, L]
        anomaly_map = anomaly_score.reshape(B, self.grid_size[0], self.grid_size[1]).unsqueeze(1)
        return anomaly_map

    def forward(self, images):
        # patch_tokens :torch.Tensor[B,L,D]
        _, patch_tokens = self.encode_image(images)

        if patch_tokens is None:
            raise RuntimeError(
                'Vision tower did not return patch tokens. '
                'Please set output_tokens=True when building VisionTransformer.'
            )

        anomaly_map = self.calculate_textual_anomaly_score(patch_tokens)
        anomaly_map = F.interpolate(
            anomaly_map,
            size=(self.out_size_h, self.out_size_w),
            mode='bilinear',
            align_corners=False,
        )

        am_np = anomaly_map.squeeze(1).cpu().numpy()
        am_np_list = [am_np[i] for i in range(am_np.shape[0])]
        return am_np_list

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()