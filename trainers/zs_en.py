import torch

from dassl.engine import TRAINER_REGISTRY

from clip import clip

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES_SELECT
from .zsclip import ZeroshotCLIP

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


@TRAINER_REGISTRY.register()
class ZeroshotEn(ZeroshotCLIP):
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Load resnet-50
        cfg.MODEL.BACKBONE.NAME = "RN50"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_rn50 = load_clip_to_cpu(cfg)
        clip_model_rn50.to(self.device)

        for params in clip_model_rn50.parameters():
            params.requires_grad_(False)

        # Load resnet-101
        cfg.MODEL.BACKBONE.NAME = "RN101"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_rn101 = load_clip_to_cpu(cfg)
        clip_model_rn101.to(self.device)

        for params in clip_model_rn101.parameters():
            params.requires_grad_(False)

        # Load vit-32
        cfg.MODEL.BACKBONE.NAME = "ViT-B/32"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_vit32 = load_clip_to_cpu(cfg)
        clip_model_vit32.to(self.device)

        for params in clip_model_vit32.parameters():
            params.requires_grad_(False)

        # Load vit-16
        cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model_vit16 = load_clip_to_cpu(cfg)
        clip_model_vit16.to(self.device)

        for params in clip_model_vit16.parameters():
            params.requires_grad_(False)

        # Custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates = [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features_rn50 = 0
        for _, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model_rn50.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_rn50 = mean_text_features_rn50 + text_features
        mean_text_features_rn50 = mean_text_features_rn50 / num_temp
        mean_text_features_rn50 = mean_text_features_rn50 / mean_text_features_rn50.norm(dim=-1, keepdim=True)

        mean_text_features_rn101 = 0
        for _, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model_rn101.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_rn101 = mean_text_features_rn101 + text_features
        mean_text_features_rn101 = mean_text_features_rn101 / num_temp
        mean_text_features_rn101 = mean_text_features_rn101 / mean_text_features_rn101.norm(dim=-1, keepdim=True)

        mean_text_features_vit32 = 0
        for _, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model_vit32.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_vit32 = mean_text_features_vit32 + text_features
        mean_text_features_vit32 = mean_text_features_vit32 / num_temp
        mean_text_features_vit32 = mean_text_features_vit32 / mean_text_features_vit32.norm(dim=-1, keepdim=True)

        mean_text_features_vit16 = 0
        for _, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model_vit16.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_vit16 = mean_text_features_vit16 + text_features
        mean_text_features_vit16 = mean_text_features_vit16 / num_temp
        mean_text_features_vit16 = mean_text_features_vit16 / mean_text_features_vit16.norm(dim=-1, keepdim=True)

        self.text_features_rn50 = mean_text_features_rn50
        self.text_features_rn101 = mean_text_features_rn101
        self.text_features_vit32 = mean_text_features_vit32
        self.text_features_vit16 = mean_text_features_vit16

        self.clip_model_rn50 = clip_model_rn50
        self.clip_model_rn101 = clip_model_rn101
        self.clip_model_vit32 = clip_model_vit32
        self.clip_model_vit16 = clip_model_vit16

    def model_inference(self, image):
        # ResNet-50
        image_features_rn50 = self.clip_model_rn50.encode_image(image)
        image_features_rn50 = image_features_rn50 / image_features_rn50.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_rn50.logit_scale.exp()
        logits_rn50 = logit_scale * image_features_rn50 @ self.text_features_rn50.t()

        # ResNet-101
        image_features_rn101 = self.clip_model_rn101.encode_image(image)
        image_features_rn101 = image_features_rn101 / image_features_rn101.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_rn101.logit_scale.exp()
        logits_rn101 = logit_scale * image_features_rn101 @ self.text_features_rn101.t()

        # Vit-32
        image_features_vit32 = self.clip_model_vit32.encode_image(image)
        image_features_vit32 = image_features_vit32 / image_features_vit32.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_vit32.logit_scale.exp()
        logits_vit32 = logit_scale * image_features_vit32 @ self.text_features_vit32.t()

        # Vit-16
        image_features_vit16 = self.clip_model_vit16.encode_image(image)
        image_features_vit16 = image_features_vit16 / image_features_vit16.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_vit16.logit_scale.exp()
        logits_vit16 = logit_scale * image_features_vit16 @ self.text_features_vit16.t()

        logits = logits_selection(logits_rn50, logits_rn101, logits_vit32, logits_vit16)

        return logits


def logits_selection(
    logits_rn50,
    logits_rn101,
    logits_vit32,
    logits_vit16):

    logits_rn50 = torch.softmax(logits_rn50, dim=-1)
    logits_rn101 = torch.softmax(logits_rn101, dim=-1)
    logits_vit32 = torch.softmax(logits_vit32, dim=-1)
    logits_vit16 = torch.softmax(logits_vit16, dim=-1)

    # Get the highest probability for each model
    max_probs_rn50, _ = torch.max(logits_rn50, dim=-1)  # bsx1
    max_probs_rn101, _ = torch.max(logits_rn101, dim=-1)
    max_probs_vit32, _ = torch.max(logits_vit32, dim=-1)

    # Weights computation
    weights = torch.cat([
        max_probs_rn50.unsqueeze(1),
        max_probs_rn101.unsqueeze(1),
        max_probs_vit32.unsqueeze(1)
    ], dim=1)  # bsx3

    weights = torch.softmax(weights, dim=1)  # bsx3

    # Initialize final logits
    logits = torch.zeros_like(logits_rn50)

    # Batch-wise logit computation
    for i in range(logits_rn50.shape[0]):
        logits[i] = torch.mean(
            torch.stack(
                [weights[i][0] * logits_rn50[i],
                 weights[i][1] * logits_rn101[i],
                 weights[i][2] * logits_vit32[i],
                 logits_vit16[i]]),
            dim=0)

    return logits
