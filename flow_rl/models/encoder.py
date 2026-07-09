"""
Shared observation encoder (ResNet-18 vision encoder + state projection).

RGBEncoder and SpatialLearnedEmbeddings are ported verbatim from jflow_match
(bc_flowmatch/models/image_processing.py): ResNet-18 with BatchNorm->GroupNorm,
features extracted at layer4, then learned spatial embeddings + bottleneck MLP.
"""

import math
from typing import Sequence, Callable

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from eqxvision.models import resnet18, resnet34, resnet50, ResNetFeatureExtractor
from eqxvision.norm_utils import replace_norm
from eqxvision.utils import CLASSIFICATION_URLS



class RGBEncoder(eqx.Module):
    """ResNet (GroupNorm) feature extractor, output at layer4 -> (C, H, W)."""

    feature_extractor: ResNetFeatureExtractor

    def __init__(self, rgb_encoder_model: str = "resnet-18", pretrained: bool = True):
        if "50" in rgb_encoder_model:
            model = resnet50(torch_weights = CLASSIFICATION_URLS["resnet50"] if pretrained else None)
        elif "34" in rgb_encoder_model:
            model = resnet34(torch_weights = CLASSIFICATION_URLS["resnet34"] if pretrained else None)
        else:
            model = resnet18(torch_weights = CLASSIFICATION_URLS["resnet18"] if pretrained else None)

        model = replace_norm(model, target = "groupnorm")
        self.feature_extractor = ResNetFeatureExtractor(model, "layer4", False)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        return self.feature_extractor(x = x, key = key)


class SpatialLearnedEmbeddings(eqx.Module):
    """Learned spatial soft-attention pooling + bottleneck MLP (jflow_match V2)."""

    kernel: Array
    bottleneck: Sequence[eqx.nn.Linear | Callable] | None
    height: int
    width: int
    channel: int
    num_features: int

    def __init__(
        self,
        height: int,
        width: int,
        channel: int,
        key: PRNGKeyArray,
        bottleneck_dim: int | None = None,
        num_features: int = 8,
    ):
        self.height = height
        self.width = width
        self.channel = channel
        self.num_features = num_features

        key, kernel_key = jax.random.split(key)
        kaiming_init = jax.nn.initializers.kaiming_normal()
        self.kernel = kaiming_init(kernel_key, (channel, height, width, num_features), dtype = jnp.float32)

        if bottleneck_dim is not None:
            k1, k2 = jax.random.split(key, 2)
            self.bottleneck = [
                eqx.nn.Linear(channel * num_features, bottleneck_dim, key = k1),
                jax.nn.gelu,
                eqx.nn.Linear(bottleneck_dim, bottleneck_dim, key = k2),
            ]
        else:
            self.bottleneck = None

    def __call__(self, features: Array) -> Array:
        # features: (C, H, W)
        features_expanded = jnp.expand_dims(features, axis = -1)  # (C, H, W, 1)
        features = jnp.sum(features_expanded * self.kernel, axis = (1, 2))  # (C, num_features)
        features = features.reshape(-1)  # (C * num_features,)
        if self.bottleneck is not None:
            for layer in self.bottleneck:
                features = layer(features)
        return features


class ObsEncoder(eqx.Module):
    """
    Encodes a (multi-step) observation into a sequence of tokens.

    Produces (encoder_num + 1) tokens of dim hidden_size (single frame, no history):
    one image token per camera + one state token. No positional embeddings are
    added to the observation tokens -- with a single frame there is no temporal
    ordering, and the tokens are already content-distinct (token roles are marked
    by the type embeddings in the policy / critic).
    """

    rgb_encoders: Sequence[RGBEncoder]
    spatial_embs: Sequence[SpatialLearnedEmbeddings]
    state_proj: eqx.nn.Linear

    hidden_size: int
    encoder_num: int
    state_dim: int
    num_tokens: int
    use_image: bool

    def __init__(
        self,
        state_dim: int = 2,
        encoder_num: int = 1,
        hidden_size: int = 512,
        img_size: int = 96,
        rgb_encoder_model: str = "resnet-18",
        pretrained: bool = True,
        num_features: int = 8,
        use_image: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.encoder_num = encoder_num
        self.state_dim = state_dim
        self.use_image = use_image
        # state-based config: vision encoder stripped, tokens = just the state token
        self.num_tokens = (encoder_num + 1) if use_image else 1

        keys = jax.random.split(key, encoder_num + 2)

        if use_image:
            self.rgb_encoders = [
                RGBEncoder(rgb_encoder_model = rgb_encoder_model, pretrained = pretrained)
                for _ in range(encoder_num)
            ]
            # ResNet-18/34 layer4 -> 512 channels; downsamples input by 32x.
            channel = 2048 if "50" in rgb_encoder_model else 512
            feat = math.ceil(img_size / 32)
            self.spatial_embs = [
                SpatialLearnedEmbeddings(
                    height = feat,
                    width = feat,
                    channel = channel,
                    key = keys[i],
                    bottleneck_dim = hidden_size,
                    num_features = num_features,
                )
                for i in range(encoder_num)
            ]
        else:
            self.rgb_encoders = []
            self.spatial_embs = []

        self.state_proj = eqx.nn.Linear(state_dim, hidden_size, key = keys[encoder_num])

    def __call__(self, img: Array, state: Array, key: PRNGKeyArray) -> Array:
        """
        Single-frame observation (no history).

        Args:
            img: (K, C, H, W) image observations (ignored / None when use_image is False)
            state: (state_dim,) low-dim state
            key: rng key (threaded to ResNet encoders)
        Returns:
            obs_tokens: (num_tokens, hidden_size) -- K image tokens + 1 state token (image),
            or just the 1 state token (state).
        """
        state_token = self.state_proj(state)[None, :]            # (1, hidden)
        if not self.use_image:
            return state_token
        rgb_keys = jax.random.split(key, img.shape[0])
        img_tokens = [
            self.spatial_embs[i](self.rgb_encoders[i](img[i], rgb_keys[i]))
            for i in range(len(self.rgb_encoders))
        ]
        img_tokens = jnp.stack(img_tokens, axis = 0)             # (K, hidden)
        return jnp.concatenate([img_tokens, state_token], axis = 0)   # (K + 1, hidden)
