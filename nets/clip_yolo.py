"""Standalone YOLOv12 + frozen CLIP weather gate baseline."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.ultralytics.yolo11_wrapper import yolo_ultralytics


class FrozenCLIPWeatherGate(nn.Module):
    """Frozen CLIP weather scorer that returns a per-image gate scale."""

    def __init__(self, model_name='ViT-B/32', prompts=None, negative_prompt_index=0,
                 strength=0.25, device='cuda', cache_dir='', weights_path=''):
        super().__init__()
        prompts = prompts or [
            'a clear traffic scene',
            'a foggy traffic scene',
            'a rainy traffic scene',
            'a dark low-light traffic scene',
        ]
        if not prompts:
            raise ValueError('CLIP weather prompts must not be empty.')
        if negative_prompt_index < 0 or negative_prompt_index >= len(prompts):
            raise ValueError(
                f'negative_prompt_index must be in [0, {len(prompts) - 1}], got {negative_prompt_index}'
            )

        self.model_name = model_name
        self.prompts = prompts
        self.negative_prompt_index = negative_prompt_index
        self.strength = float(strength)
        self.clip_device = torch.device(device if device == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.backend = 'openai'
        self.clip = None
        self.tokenizer = None
        load_name = weights_path or model_name

        if weights_path and not os.path.exists(weights_path):
            raise FileNotFoundError(f'CLIP_WEIGHTS_PATH does not exist: {weights_path}')

        if weights_path and os.path.isdir(weights_path):
            self.backend = 'transformers'
            model_dir = os.path.join(weights_path, '0_CLIPModel')
            if not os.path.isdir(model_dir):
                model_dir = weights_path
            try:
                from transformers import CLIPModel, CLIPTokenizer
            except ImportError as exc:
                raise ImportError(
                    'Directory CLIP_WEIGHTS_PATH requires transformers. '
                    'Install transformers or use an OpenAI CLIP .pt weight file.'
                ) from exc
            self.model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(self.clip_device)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_dir, local_files_only=True)
        else:
            try:
                import clip
            except ImportError as exc:
                raise ImportError(
                    'FrozenCLIPWeatherGate requires the OpenAI CLIP package. '
                    'Install it before running CLIP experiments.'
                ) from exc
            self.clip = clip
            kwargs = {'device': self.clip_device}
            if cache_dir:
                kwargs['download_root'] = cache_dir
            self.model, _ = clip.load(load_name, **kwargs)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            if self.backend == 'transformers':
                text_tokens = self.tokenizer(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors='pt',
                ).to(self.clip_device)
                text_features = self.model.get_text_features(**text_tokens).float()
            else:
                text_tokens = self.clip.tokenize(prompts, truncate=True).to(self.clip_device)
                text_features = self.model.encode_text(text_tokens).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.register_buffer('text_features', text_features, persistent=False)
        print(f'[FrozenCLIPWeatherGate] Loaded {load_name} with {len(prompts)} prompts on {self.clip_device}.')

    def _prepare_images(self, images):
        x = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        x = x.clamp(0.0, 1.0)
        mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def forward(self, images):
        original_device = images.device
        active_device = self.text_features.device
        clip_images = self._prepare_images(images).to(active_device)
        if self.backend == 'transformers':
            image_features = self.model.get_image_features(pixel_values=clip_images).float()
        else:
            image_features = self.model.encode_image(clip_images).float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        logits = 100.0 * image_features @ self.text_features.t()
        probs = logits.softmax(dim=-1)
        clean_prob = probs[:, self.negative_prompt_index:self.negative_prompt_index + 1]
        adverse_prob = (1.0 - clean_prob).clamp(0.0, 1.0)
        scale = 1.0 + self.strength * adverse_prob
        return {
            'probs': probs.to(original_device),
            'adverse_prob': adverse_prob.to(original_device),
            'scale': scale.to(original_device),
        }


class YOLOv12CLIPWeatherGate(nn.Module):
    """YOLOv12 detector with a single frozen CLIP weather gate on neck features."""

    def __init__(self, num_classes, yolo_pretrained_path='model_data/yolo12n_Ultralytics.pt',
                 clip_model_name='ViT-B/32', clip_prompts=None, clip_negative_prompt_index=0,
                 clip_strength=0.25, clip_device='cuda', clip_cache_dir='', clip_weights_path=''):
        super().__init__()
        self.detector = yolo_ultralytics(num_cls=num_classes, pretrained=yolo_pretrained_path)
        self.clip_gate = FrozenCLIPWeatherGate(
            model_name=clip_model_name,
            prompts=clip_prompts,
            negative_prompt_index=clip_negative_prompt_index,
            strength=clip_strength,
            device=clip_device,
            cache_dir=clip_cache_dir,
            weights_path=clip_weights_path,
        )
        self.last_clip_adverse_mean = None

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if key.startswith('clip_gate.model.'):
                del state[key]
        return state

    def forward(self, images, return_features=False, raw_output=False, return_clip=False):
        clip_outputs = self.clip_gate(images)
        clip_scale = clip_outputs['scale'].view(-1, 1, 1, 1)
        _, neck_feats = self.detector.forward_backbone_and_neck(images)
        gated_feats = [feat * clip_scale for feat in neck_feats]
        predictions = self.detector.forward_from_features(gated_feats, raw_output=raw_output)
        self.last_clip_adverse_mean = clip_outputs['adverse_prob'].detach().mean()
        if return_clip:
            return predictions, clip_outputs
        if return_features:
            return predictions, tuple(gated_feats)
        return predictions
