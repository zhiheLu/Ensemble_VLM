import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_checkpoint

from clip import clip

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES_SELECT


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


class CustomCLIP(nn.Module):
    def __init__(self, 
                 classnames, 
                 clip_models, 
                 templates,
                 num_temp,
                 downscale=32,
                 num_weights=4):
        super().__init__()
        self.templates = templates
        self.dtype = clip_models[0].dtype

        mean_text_features_rn50 = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            text_features = clip_models[0].encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_rn50 = mean_text_features_rn50 + text_features
        mean_text_features_rn50 = mean_text_features_rn50 / num_temp
        mean_text_features_rn50 = mean_text_features_rn50 / mean_text_features_rn50.norm(dim=-1, keepdim=True)

        mean_text_features_rn101 = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            text_features = clip_models[1].encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_rn101 = mean_text_features_rn101 + text_features
        mean_text_features_rn101 = mean_text_features_rn101 / num_temp
        mean_text_features_rn101 = mean_text_features_rn101 / mean_text_features_rn101.norm(dim=-1, keepdim=True)

        mean_text_features_vit32 = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            text_features = clip_models[2].encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_vit32 = mean_text_features_vit32 + text_features
        mean_text_features_vit32 = mean_text_features_vit32 / num_temp
        mean_text_features_vit32 = mean_text_features_vit32 / mean_text_features_vit32.norm(dim=-1, keepdim=True)

        mean_text_features_vit16 = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            text_features = clip_models[3].encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features_vit16 = mean_text_features_vit16 + text_features
        mean_text_features_vit16 = mean_text_features_vit16 / num_temp
        mean_text_features_vit16 = mean_text_features_vit16 / mean_text_features_vit16.norm(dim=-1, keepdim=True)

        self.text_features_rn50 = mean_text_features_rn50
        self.text_features_rn101 = mean_text_features_rn101
        self.text_features_vit32 = mean_text_features_vit32
        self.text_features_vit16 = mean_text_features_vit16

        self.clip_model_rn50 = clip_models[0]
        self.clip_model_rn101 = clip_models[1]
        self.clip_model_vit32 = clip_models[2]
        self.clip_model_vit16 = clip_models[3]

        # Compute the dimension of model input
        feature_dim = [
            self.text_features_rn50.shape[-1],
            self.text_features_rn101.shape[-1],
            self.text_features_vit32.shape[-1],
            self.text_features_vit16.shape[-1]
        ]
        feature_dim_sum = sum(feature_dim)

        # Learnable weights
        self.learn_module = nn.Sequential(
            nn.Linear(feature_dim_sum, feature_dim_sum//downscale),
            nn.ReLU(),
            nn.Linear(feature_dim_sum//downscale, num_weights)
        ).type(self.dtype)

    def forward(self, image):

        # Feature extraction
        with torch.no_grad():
            image_features_rn50 = self.clip_model_rn50.encode_image(image.type(self.dtype))
            image_features_rn101 = self.clip_model_rn101.encode_image(image.type(self.dtype))
            image_features_vit32 = self.clip_model_vit32.encode_image(image.type(self.dtype))
            image_features_vit16 = self.clip_model_vit16.encode_image(image.type(self.dtype))
        
        # Weight generation
        feat_cat = torch.cat([
            image_features_rn50, 
            image_features_rn101, 
            image_features_vit32, 
            image_features_vit16
        ], dim=1)  # (B, C)
        weights = self.learn_module(feat_cat).unsqueeze(2)  # (B, 4, 1)
        weights = torch.softmax(weights, dim=1)  # (B, 4, 1)

        # Logit prediction
        # ResNet-50
        image_features_rn50 = image_features_rn50 / image_features_rn50.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_rn50.logit_scale.exp()
        logits_rn50 = logit_scale * image_features_rn50 @ self.text_features_rn50.t()

        # ResNet-101
        image_features_rn101 = image_features_rn101 / image_features_rn101.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_rn101.logit_scale.exp()
        logits_rn101 = logit_scale * image_features_rn101 @ self.text_features_rn101.t()

        # Vit-32
        image_features_vit32 = image_features_vit32 / image_features_vit32.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_vit32.logit_scale.exp()
        logits_vit32 = logit_scale * image_features_vit32 @ self.text_features_vit32.t()

        # Vit-16
        image_features_vit16 = image_features_vit16 / image_features_vit16.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model_vit16.logit_scale.exp()
        logits_vit16 = logit_scale * image_features_vit16 @ self.text_features_vit16.t()

        logits_rn50 = torch.softmax(logits_rn50, dim=-1)
        logits_rn101 = torch.softmax(logits_rn101, dim=-1)
        logits_vit32 = torch.softmax(logits_vit32, dim=-1)
        logits_vit16 = torch.softmax(logits_vit16, dim=-1)
        
        # Learnable weights
        logits_cat = torch.cat([
            logits_rn50.unsqueeze(1), 
            logits_rn101.unsqueeze(1), 
            logits_vit32.unsqueeze(1), 
            logits_vit16.unsqueeze(1)
        ], dim=1)  # (B, 4, C)

        logits_final = torch.mean(logits_cat * weights, dim=1)  # (B, C)

        return logits_final


@TRAINER_REGISTRY.register()
class TuneEn(TrainerX):
    """
     Learning the weights for logits of each model.
    """
    templates = IMAGENET_TEMPLATES_SELECT

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ENLEARN.PREC in ["fp16", "fp32", "amp"]

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
            self.templates = [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")
        
        if cfg.TRAINER.ENLEARN.PREC == "fp32" or cfg.TRAINER.ENLEARN.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model_rn50.float()
            clip_model_rn101.float()
            clip_model_vit32.float()
            clip_model_vit16.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(
            classnames, 
            [clip_model_rn50, clip_model_rn101, clip_model_vit32, clip_model_vit16],
            self.templates,
            num_temp,
            downscale=cfg.TRAINER.ENLEARN.DOWNSCALE,
            num_weights=cfg.TRAINER.ENLEARN.NUM_WEIGHT
        )

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "learn_module" not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")

        self.model.to(self.device)

        # NOTE: only give weight generator to the optimizer
        self.optim = build_optimizer(self.model.learn_module, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("learn_module", self.model.learn_module, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ENLEARN.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.ENLEARN.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            out = self.model(image)
            loss = F.cross_entropy(out, label)

            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)