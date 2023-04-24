#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

from vc_models import vc_models_dir_path
from omegaconf import OmegaConf
from PIL import Image
import os
import json
import hydra
import torch, torchvision.transforms as T
import numpy as np
from pathlib import Path
from voltron.models import VMVP, VR3M, VRN3M, VCond, VDual, VGen
from transformers import Trainer


# ===================================
# Model Loading
# ===================================

NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class VCondModel(torch.nn.Module):
    def __init__(self, vcond, vector_extractor):
        super().__init__()
        self.vcond = vcond
        self.vector_extractor = vector_extractor

    def forward(self, img, mode="visual"):
        multimodal_embeddings = self.vcond(img, mode=mode)
        representation = self.vector_extractor(multimodal_embeddings)
        return representation


class VMVPModel(torch.nn.Module):
    def __init__(self, vmvp):
        super().__init__()
        self.vmvp = vmvp

    def forward(self, img, mode="cls"):
        # mode is either {"cls", "patch"}
        representation = self.vmvp.get_representations(img, mode=mode).squeeze()
        return representation


def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
    """
    Load the pretrained model based on the config corresponding to the embedding_name
    """

    cache_dir = (
        "/data/anthony/object_centric_vp/eai-vc/cortexbench/mujoco_vc/visual_imitation/"
    )

    def fn(input):
        return transforms(Image.fromarray(input)).unsqueeze(0)

    def final_transforms(transforms):
        if input_type == np.ndarray:
            return fn
        else:
            return transforms

    if embedding_name in ["r-mvp", "v-cond"]:
        print(f"Loading {embedding_name} model")

        from voltron import instantiate_extractor, load

        device = "cuda"
        model, preprocess = load(
            embedding_name,
            device=device,
            freeze=True,
            cache=cache_dir,
        )

        embedding_dim = 384

        if embedding_name == "v-cond":
            vector_extractor = instantiate_extractor(model)()
            final_model = VCondModel(model, vector_extractor).to(device)
        elif embedding_name == "r-mvp":
            final_model = VMVPModel(model).to(device)
        else:
            raise NotImplementedError

        final_model.eval()

        transforms = preprocess
        transforms.transforms.insert(2, T.ToTensor())

        return final_model, embedding_dim, final_transforms(transforms), None
    elif embedding_name == "v-mvp-ego_objects":

        config_path = Path(cache_dir) / "r-mvp/r-mvp-config.json"
        with open(config_path, "r") as f:
            model_kwargs = json.load(f)

        cfg = OmegaConf.load(
            "/data/anthony/object_centric_vp/results/ego_objects_r-mvp/config.json"
        )

        embedding_dim = 384

        model = VMVP(
            resolution=cfg.dataset.resolution,
            patch_size=cfg.model.patch_size,
            encoder_depth=cfg.model.encoder_depth,
            encoder_embed_dim=cfg.model.encoder_embed_dim,
            encoder_n_heads=cfg.model.encoder_n_heads,
            decoder_depth=cfg.model.decoder_depth,
            decoder_embed_dim=cfg.model.decoder_embed_dim,
            decoder_n_heads=cfg.model.decoder_n_heads,
            optimizer=cfg.model.optimizer,
            schedule=cfg.model.schedule,
            base_lr=cfg.model.base_lr,
            min_lr=cfg.model.min_lr,
            effective_bsz=cfg.model.effective_bsz,
            betas=cfg.model.betas,
            weight_decay=cfg.model.weight_decay,
            warmup_epochs=cfg.dataset.warmup_epochs,
            max_epochs=cfg.dataset.max_epochs,
            mlp_ratio=cfg.model.mlp_ratio,
            norm_pixel_loss=cfg.model.norm_pixel_loss,
        )
        trainer = Trainer(model)
        trainer._load_from_checkpoint(
            "/data/anthony/object_centric_vp/results/ego_objects_r-mvp/checkpoint-146000",
            model,
        )
        print("loaded model from checkpoint")

        model = VMVPModel(model)
        model.eval()
        preprocess = T.Compose(
            [
                T.Resize(model_kwargs["resolution"]),
                T.CenterCrop(model_kwargs["resolution"]),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=NORMALIZATION[0], std=NORMALIZATION[1]),
            ]
        )
        transforms = preprocess

        return model, embedding_dim, final_transforms(transforms), None
    else:
        config_path = os.path.join(
            vc_models_dir_path, "conf/model", embedding_name + ".yaml"
        )
        print("Loading config path: %s" % config_path)
        config = OmegaConf.load(config_path)
        model, embedding_dim, transforms, metadata = hydra.utils.call(config)

        model = (
            model.eval()
        )  # model loading API is unreliable, call eval to be double sure

        def final_transforms(transforms):
            if input_type == np.ndarray:
                return lambda input: transforms(Image.fromarray(input)).unsqueeze(0)
            else:
                return transforms

        return model, embedding_dim, final_transforms(transforms), metadata


# ===================================
# Temporal Embedding Fusion
# ===================================
def fuse_embeddings_concat(embeddings: list):
    assert type(embeddings[0]) == np.ndarray
    return np.array(embeddings).ravel()


def fuse_embeddings_flare(embeddings: list):
    if type(embeddings[0]) == np.ndarray:
        history_window = len(embeddings)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1].copy())
        return np.array(delta).ravel()
    elif type(embeddings[0]) == torch.Tensor:
        history_window = len(embeddings)
        # each embedding will be (Batch, Dim)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1])
        return torch.cat(delta, dim=1)
    else:
        print("Unsupported embedding format in fuse_embeddings_flare.")
        print("Provide either numpy.ndarray or torch.Tensor.")
        quit()
