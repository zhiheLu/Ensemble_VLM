from tqdm import tqdm
import torch
import torch.nn as nn

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
class TrainingfreeEn(ZeroshotCLIP):
    """
        Training-free ensemble by greedy search.
    """
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

        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            # self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]
            self.templates = [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features_rn50 = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model_rn50.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_rn50 = mean_text_features_rn50 + text_features
        mean_text_features_rn50 = mean_text_features_rn50 / num_temp
        mean_text_features_rn50 = mean_text_features_rn50 / mean_text_features_rn50.norm(dim=-1, keepdim=True)

        mean_text_features_rn101 = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model_rn101.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_rn101 = mean_text_features_rn101 + text_features
        mean_text_features_rn101 = mean_text_features_rn101 / num_temp
        mean_text_features_rn101 = mean_text_features_rn101 / mean_text_features_rn101.norm(dim=-1, keepdim=True)

        mean_text_features_vit32 = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model_vit32.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_vit32 = mean_text_features_vit32 + text_features
        mean_text_features_vit32 = mean_text_features_vit32 / num_temp
        mean_text_features_vit32 = mean_text_features_vit32 / mean_text_features_vit32.norm(dim=-1, keepdim=True)

        mean_text_features_vit16 = 0
        for i, temp in enumerate(self.templates):
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

    def model_inference(self, image, w1, w2, w3):
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

        logits_rn50 = torch.softmax(logits_rn50, dim=-1)
        logits_rn101 = torch.softmax(logits_rn101, dim=-1)
        logits_vit32 = torch.softmax(logits_vit32, dim=-1)
        logits_vit16 = torch.softmax(logits_vit16, dim=-1)

        logits = (w1 * logits_rn50 + w2 * logits_rn101 + w3 * logits_vit32 + logits_vit16) / 4.0

        return logits
    
    @torch.no_grad()
    def test_search(self, split=None, w1=0.0, w2=0.0, w3=0.0):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        elif split == "train":
            data_loader = self.train_loader_x
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for _, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input, w1, w2, w3)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        if split == "train":
            return results['accuracy'], [w1, w2, w3]
        else:
            return list(results.values())[0]

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

