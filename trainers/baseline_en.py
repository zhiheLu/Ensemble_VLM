import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_checkpoint

from .cocoop import CoCoOp
from .promptsrc import PromptSRC

@TRAINER_REGISTRY.register()
class CoCoOpEn(TrainerX):
    def build_model(self):
        cfg = self.cfg

        # Load resnet-50
        cfg.MODEL.BACKBONE.NAME = "RN50"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.cocoop_rn50 = CoCoOp(cfg)

        self.dtype = self.cocoop_rn50.model.dtype
        
        self.cocoop_rn50.load_model(
            f"{cfg.TRAINER.ENLEARN.MODEL_DIR}/{cfg.MODEL.BACKBONE.NAME}/seed{cfg.SEED}", 
            epoch=10
        )
        self.cocoop_rn50.model.eval()

        # Load resnet-101
        cfg.MODEL.BACKBONE.NAME = "RN101"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.cocoop_rn101 = CoCoOp(cfg)

        self.cocoop_rn101.load_model(
            f"{cfg.TRAINER.ENLEARN.MODEL_DIR}/{cfg.MODEL.BACKBONE.NAME}/seed{cfg.SEED}", 
            epoch=10
        )
        self.cocoop_rn101.model.eval()

        # Load vit-32
        cfg.MODEL.BACKBONE.NAME = "ViT-B/32"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.cocoop_vit32 = CoCoOp(cfg)
        
        self.cocoop_vit32.load_model(
            f"{cfg.TRAINER.ENLEARN.MODEL_DIR}/{cfg.MODEL.BACKBONE.NAME}/seed{cfg.SEED}", 
            epoch=10
        )
        self.cocoop_vit32.model.eval()

        # Load vit-16
        cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.cocoop_vit16 = CoCoOp(cfg)
        
        self.cocoop_vit16.load_model(
            f"{cfg.TRAINER.ENLEARN.MODEL_DIR}/{cfg.MODEL.BACKBONE.NAME}/seed{cfg.SEED}", 
            epoch=10
        )
        self.cocoop_vit16.model.eval()

        # Weight generator
        self.model = nn.Sequential(
            nn.Linear(2560, 2560//cfg.TRAINER.ENLEARN.DOWNSCALE),
            nn.ReLU(),
            nn.Linear(2560//cfg.TRAINER.ENLEARN.DOWNSCALE, cfg.TRAINER.ENLEARN.NUM_WEIGHT)
        ).type(self.dtype)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        with torch.no_grad():
            logits_rn50, feat_rn50 = self.cocoop_rn50.model_inference(image, feature=True)
            logits_rn101, feat_rn101 = self.cocoop_rn101.model_inference(image, feature=True)
            logits_vit32, feat_vit32 = self.cocoop_vit32.model_inference(image, feature=True)
            logits_vit16, feat_vit16 = self.cocoop_vit16.model_inference(image, feature=True)

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                # Weight generation
                feat_list = [feat_rn50, feat_rn101, feat_vit32, feat_vit16]
                feat_cat = torch.cat(feat_list, dim=1)
                weights = model(feat_cat).unsqueeze(2)  # (B, N, 1)

                logits_cat = torch.cat(
                    [logits_rn50.unsqueeze(1), 
                    logits_rn101.unsqueeze(1), 
                    logits_vit32.unsqueeze(1), 
                    logits_vit16.unsqueeze(1)], dim=1)  # (B, 4, 500)

                logits = torch.sum(logits_cat * weights, dim=1)
                loss = F.cross_entropy(logits, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            # Weight generation
            feat_list = [feat_rn50, feat_rn101, feat_vit32, feat_vit16]
            feat_cat = torch.cat(feat_list, dim=1)
            weights = model(feat_cat).unsqueeze(2)  # (B, N, 1)

            logits_cat = torch.cat(
                [logits_rn50.unsqueeze(1), 
                 logits_rn101.unsqueeze(1), 
                 logits_vit32.unsqueeze(1), 
                 logits_vit16.unsqueeze(1)], dim=1)  # (B, 4, 500)

            logits = torch.sum(logits_cat * weights, dim=1)
            loss = F.cross_entropy(logits, label)

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
    
    def model_inference(self, image):
        with torch.no_grad():
            logits_rn50, feat_rn50 = self.cocoop_rn50.model_inference(image, feature=True)
            logits_rn101, feat_rn101 = self.cocoop_rn101.model_inference(image, feature=True)
            logits_vit32, feat_vit32 = self.cocoop_vit32.model_inference(image, feature=True)
            logits_vit16, feat_vit16 = self.cocoop_vit16.model_inference(image, feature=True)
        
        # Weight generation
        feat_list = [feat_rn50, feat_rn101, feat_vit32, feat_vit16]
        feat_cat = torch.cat(feat_list, dim=1)
        weights = self.model(feat_cat).unsqueeze(2)  # (B, N, 1)

        logits_cat = torch.cat(
            [logits_rn50.unsqueeze(1), 
             logits_rn101.unsqueeze(1), 
             logits_vit32.unsqueeze(1), 
             logits_vit16.unsqueeze(1)], dim=1)  # (B, 4, 500)

        logits = torch.sum(logits_cat * weights, dim=1)

        return logits
    
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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


@TRAINER_REGISTRY.register()
class PromptSRCEn(TrainerX):
    def build_model(self):
        cfg = self.cfg

        # Load vit-32
        cfg.MODEL.BACKBONE.NAME = "ViT-B/32"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.promptsrc_vit32 = PromptSRC(cfg)

        self.dtype = self.promptsrc_vit32.model.dtype
        
        self.promptsrc_vit32.load_model(
            f"{cfg.TRAINER.ENLEARN.MODEL_DIR}/{cfg.MODEL.BACKBONE.NAME}/seed{cfg.SEED}", 
            epoch=20
        )
        self.promptsrc_vit32.model.eval()

        # Load vit-16
        cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        self.promptsrc_vit16 = PromptSRC(cfg)
        
        self.promptsrc_vit16.load_model(
            f"{cfg.TRAINER.ENLEARN.MODEL_DIR}/{cfg.MODEL.BACKBONE.NAME}/seed{cfg.SEED}", 
            epoch=20
        )
        self.promptsrc_vit16.model.eval()

        # Weight generator
        self.model = nn.Sequential(
            nn.Linear(1024, 1024//cfg.TRAINER.ENLEARN.DOWNSCALE),
            nn.ReLU(),
            nn.Linear(1024//cfg.TRAINER.ENLEARN.DOWNSCALE, cfg.TRAINER.ENLEARN.NUM_WEIGHT)
        ).type(self.dtype)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        with torch.no_grad():
            logits_vit32, feat_vit32 = self.promptsrc_vit32.model_inference(image, feature=True)
            logits_vit16, feat_vit16 = self.promptsrc_vit16.model_inference(image, feature=True)

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                # Weight generation
                feat_list = [feat_vit32, feat_vit16]
                feat_cat = torch.cat(feat_list, dim=1)
                weights = model(feat_cat).unsqueeze(2)  # (B, 2, 1)

                logits_cat = torch.cat(
                    [logits_vit32.unsqueeze(1),
                     logits_vit16.unsqueeze(1)], dim=1)  # (B, 2, 500)

                logits = torch.sum(logits_cat * weights, dim=1)
                loss = F.cross_entropy(logits, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            # Weight generation
            feat_list = [feat_vit32, feat_vit16]
            feat_cat = torch.cat(feat_list, dim=1)
            weights = model(feat_cat).unsqueeze(2)  # (B, 2, 1)

            logits_cat = torch.cat(
                [logits_vit32.unsqueeze(1),
                 logits_vit16.unsqueeze(1)], dim=1)  # (B, 2, 500)

            logits = torch.sum(logits_cat * weights, dim=1)
            loss = F.cross_entropy(logits, label)

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
    
    def model_inference(self, image):
        with torch.no_grad():
            logits_vit32, feat_vit32 = self.promptsrc_vit32.model_inference(image, feature=True)
            logits_vit16, feat_vit16 = self.promptsrc_vit16.model_inference(image, feature=True)
        
        # Weight generation
        feat_list = [feat_vit32, feat_vit16]
        feat_cat = torch.cat(feat_list, dim=1)
        weights = self.model(feat_cat).unsqueeze(2)  # (B, 2, 1)

        logits_cat = torch.cat(
            [logits_vit32.unsqueeze(1), 
             logits_vit16.unsqueeze(1)], dim=1)  # (B, 2, 500)

        logits = torch.sum(logits_cat * weights, dim=1)

        return logits
    
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
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)