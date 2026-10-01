"""
Context-Gated CBAM-Transformer Network (CG-CBAT) for natural scene recognition.

Architecture (v4, FPN-style feature fusion + multi-scale context gating):
    ResNet50 backbone, extracted at TWO stages via timm's `features_only` API:
        - mid stage   (stride 16, 1024 channels, 14x14 for 224px input)
        - final stage (stride 32, 2048 channels,  7x7 for 224px input)
    Each stage gets its own CBAM (channel attention needs per-channel-count
    MLPs; spatial attention is shared since it only ever sees a 2-channel
    pooled map regardless of the input channel count).

    Diagnosed problem with v3: the mid-stage map only reached the classifier
    INDIRECTLY, as tokens contributing to a global context vector that then
    produced a channel-wise gate on the final-stage map. That is a narrow
    bottleneck -- it can only reweight which final-stage channels matter, it
    cannot inject the spatial/textural detail the coarse 7x7 final-stage map
    has already lost by that depth. Result: marginal gain on one dataset,
    a dead wash on the other.

    v4 fix -- an actual FPN-style lateral fusion path:
        - The mid-stage CBAM-refined map (1024ch, 14x14) is projected through
          a stride-2 3x3 conv + BatchNorm to (2048ch, 7x7), i.e. real spatial
          downsampling and channel matching, not just pooling for a summary
          statistic.
        - That lateral map is added element-wise to the final-stage
          CBAM-refined map to form `fused_feat` (2048ch, 7x7), which now
          carries genuine fine-grained content from the mid stage, not just a
          reweighting signal derived from it.
        - The lateral contribution is scaled by a learnable scalar
          initialized to 0 (`lateral_scale`), so at the start of training
          `fused_feat == final_feat` exactly -- the model begins from the
          already-reasonable final-stage-only behaviour of v2/v3 and only
          gradually learns to blend in mid-stage detail, rather than
          immediately perturbing pretrained final-stage features with an
          untrained conv's output. Same spirit as the residual gate fix that
          made v1->v2 training stable.
        - The MultiScaleTransformerBranch is unchanged: it still tokenizes
          both ORIGINAL CBAM-refined maps (245 tokens total) and produces one
          global "scene context" vector via a shared Transformer encoder.
        - That global context now gates `fused_feat` (not final_feat alone),
          same residual form as v2/v3: `fused_feat * (1 + gate)`.
    So the global branch still supplies a context-conditioned reweighting
    signal, but the thing being reweighted now actually contains fused
    multi-scale spatial content, not just coarse final-stage semantics.

    v5 regularization fix (round 5): a matched-epoch validation showed v4
    losing to a plain ResNet50 on all 5 benchmark datasets, with the gap
    scaling with how few images-per-class the dataset has (worst on SUN397,
    ~30 images/class, -10.73pp). Diagnosis: the added capacity (2x CBAM,
    transformer branch, FPN lateral fusion) combined with an aggressive 10x
    LR on the new modules was overfitting in low-data-per-class regimes.
    Architecture itself (CBAM/fusion/gating) is UNCHANGED from v4 -- this
    round only adds regularization:
        - nn.Dropout(p=0.3) between the pooled feature vector and the final
          classifier (`self.dropout`, see __init__/forward).
        - dropout=0.3 explicitly set inside MultiScaleTransformerBranch's
          nn.TransformerEncoderLayer (previously left at PyTorch's default,
          0.1).
    The new-modules LR multiplier and weight decay were also changed
    (10x -> 5x, weight_decay=0.05 on new-module params) in train_eval.py's
    optimizer setup for cg_cbat_resnet50 -- not in this file.
"""

import torch
import torch.nn as nn
import timm


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class MultiScaleTransformerBranch(nn.Module):
    """Reads CNN feature maps from MULTIPLE stages (different channel counts
    and spatial resolutions) as one combined sequence of spatial tokens, and
    builds a single global context vector via self-attention across the
    whole scene, jointly attending to fine (mid-stage) and coarse
    (final-stage) spatial evidence.

    Each stage gets its own input projection (channel counts differ) and its
    own learned positional embedding (token counts/resolutions differ); the
    Transformer encoder itself is shared across the concatenated sequence.
    """

    def __init__(self, stage_channels: list, stage_num_tokens: list,
                 embed_dim: int = 256, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        assert len(stage_channels) == len(stage_num_tokens)
        self.projs = nn.ModuleList([
            nn.Linear(c, embed_dim) for c in stage_channels
        ])
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, n, embed_dim)) for n in stage_num_tokens
        ])
        for pe in self.pos_embeds:
            nn.init.trunc_normal_(pe, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 2,
            dropout=0.3, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feats: list):
        """feats: list of (B, C_i, H_i, W_i) feature maps, one per stage,
        matching the order of stage_channels/stage_num_tokens at init."""
        all_tokens = []
        for feat, proj, pos_embed in zip(feats, self.projs, self.pos_embeds):
            b, c, h, w = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            tokens = proj(tokens) + pos_embed
            all_tokens.append(tokens)
        tokens = torch.cat(all_tokens, dim=1)  # (B, sum(H_i*W_i), embed_dim)
        encoded = self.encoder(tokens)
        return encoded.mean(dim=1)  # global context: (B, embed_dim)


class ContextGatedCBATNet(nn.Module):
    """ResNet50 (multi-scale: mid stage + final stage) + per-stage CBAM +
    FPN-style lateral fusion of mid-stage detail into the final-stage map +
    multi-scale Transformer branch with context-conditioned feedback gating
    of the FUSED feature map.

    Which two stages are used is NOT hardcoded by index guesswork -- it is
    resolved dynamically from the backbone's own `feature_info` at
    construction time, by picking the stages whose stride (reduction) is 16
    (mid, more spatial detail) and 32 (final, most semantic), the highest two
    reduction factors timm's `features_only` API exposes for resnet50. This
    keeps the code correct even if `backbone_name` changes to another
    resnet-family model with a different stage layout.
    """

    MID_STRIDE = 16
    FINAL_STRIDE = 32

    def __init__(self, num_classes: int, img_size: int = 224,
                 embed_dim: int = 256, num_heads: int = 4, num_layers: int = 2,
                 backbone_name: str = "resnet50"):
        super().__init__()

        # Probe the backbone's stage layout dynamically (no hardcoded
        # out_indices) -- verified for resnet50 via feature_info at dev time:
        # reduction=[2,4,8,16,32], channels=[64,256,512,1024,2048], i.e.
        # out_indices (3, 4) => mid stage (1024ch, stride16) and final stage
        # (2048ch, stride32).
        probe = timm.create_model(backbone_name, pretrained=False, features_only=True)
        reductions = probe.feature_info.reduction()
        channels_all = probe.feature_info.channels()
        del probe

        try:
            mid_idx = reductions.index(self.MID_STRIDE)
            final_idx = reductions.index(self.FINAL_STRIDE)
        except ValueError as e:
            raise RuntimeError(
                f"Backbone '{backbone_name}' does not expose stride-"
                f"{self.MID_STRIDE}/{self.FINAL_STRIDE} stages via "
                f"features_only (reductions found: {reductions}). "
                f"Update ContextGatedCBATNet.MID_STRIDE/FINAL_STRIDE."
            ) from e

        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            out_indices=(mid_idx, final_idx),
        )
        mid_channels = channels_all[mid_idx]
        final_channels = channels_all[final_idx]
        mid_stride = reductions[mid_idx]
        final_stride = reductions[final_idx]
        mid_grid = img_size // mid_stride
        final_grid = img_size // final_stride
        mid_num_tokens = mid_grid * mid_grid
        final_num_tokens = final_grid * final_grid

        self.final_channels = final_channels

        self.cbam_mid = CBAM(mid_channels)
        self.cbam_final = CBAM(final_channels)

        # FPN-style lateral fusion: real spatial downsampling (stride-2 3x3
        # conv, not pooling) + channel projection of the mid-stage map to the
        # final stage's resolution/channel count, so it can be added
        # element-wise and actually carry spatial/textural content forward
        # (not just a summary statistic like a gate would).
        self.lateral_conv = nn.Conv2d(
            mid_channels, final_channels, kernel_size=3, stride=2, padding=1,
        )
        self.lateral_bn = nn.BatchNorm2d(final_channels)
        # Learnable scalar, initialized near-zero (LayerScale-style, e.g.
        # Touvron et al. CaiT) rather than exactly zero: at the start of
        # training fused_feat ~= final_feat (the lateral path contributes
        # almost nothing), so the model begins from v2/v3's already-
        # reasonable final-stage-only behaviour and only gradually learns to
        # blend in mid-stage detail, instead of immediately perturbing
        # pretrained final-stage features with an untrained conv's output.
        # An exact 0.0 init was deliberately avoided: by the chain rule,
        # d(loss)/d(lateral_conv.weight) is itself multiplied by
        # lateral_scale, so an exact-zero scale gives lateral_conv/lateral_bn
        # a literal zero gradient on the very first step (only lateral_scale
        # itself would move, since its own gradient does not depend on its
        # current value) -- a fragile bootstrap. Starting at 1e-4 keeps the
        # near-identity behaviour while letting every new parameter receive
        # real gradient signal from step one.
        self.lateral_scale = nn.Parameter(torch.full((1,), 1e-4))

        self.transformer = MultiScaleTransformerBranch(
            stage_channels=[mid_channels, final_channels],
            stage_num_tokens=[mid_num_tokens, final_num_tokens],
            embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim, final_channels),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # v5 regularization fix: dropout between the pooled feature vector
        # and the final classifier, to fight overfitting observed in the
        # matched-epoch validation (CG-CBAT v4 lost to plain ResNet50 on all
        # 5 datasets, worst in low-images-per-class regimes like SUN397).
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(final_channels, num_classes)

    def load_backbone_checkpoint(self, checkpoint_path: str, map_location="cpu"):
        """Round 6: load a plain timm.create_model('resnet50', ..., num_classes=N)
        checkpoint (full classifier-head model, saved as this repo's
        train_eval.py checkpoint dict with a 'model_state' key) into
        self.backbone (a features_only feature extractor built from the same
        underlying resnet50).

        Verified directly (not guessed) by comparing state_dict key sets: the
        plain model's state dict and self.backbone's state dict share IDENTICAL
        key names and tensor shapes for every backbone conv/bn parameter --
        conv1/bn1/layer1..layer4, no prefix differences -- because features_only
        wraps the exact same underlying module tree. The only keys present in
        the plain checkpoint but absent from self.backbone are the classifier
        head ('fc.weight', 'fc.bias'), which is expected and intentionally
        dropped since features_only has no classifier.

        Returns (num_matched, num_dropped) for the caller to sanity-check.
        """
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        full_state = ckpt["model_state"] if "model_state" in ckpt else ckpt

        backbone_keys = set(self.backbone.state_dict().keys())
        matched_state = {k: v for k, v in full_state.items() if k in backbone_keys}
        dropped = [k for k in full_state.keys() if k not in backbone_keys]

        missing, unexpected = self.backbone.load_state_dict(matched_state, strict=False)
        if missing:
            raise RuntimeError(
                f"load_backbone_checkpoint: {len(missing)} backbone parameters "
                f"were NOT found in checkpoint '{checkpoint_path}' -- refusing "
                f"to silently load a partial backbone. Missing (first 10): "
                f"{missing[:10]}"
            )
        assert not unexpected, f"unexpected keys after filtering: {unexpected}"

        print(
            f"[load_backbone_checkpoint] loaded {len(matched_state)}/"
            f"{len(backbone_keys)} backbone tensors from '{checkpoint_path}' "
            f"(dropped {len(dropped)} non-backbone keys: {dropped})",
            flush=True,
        )
        return len(matched_state), len(dropped)

    def freeze_backbone(self):
        """Round 6: freeze self.backbone's parameters for the entire run
        (not just a warm-up window)."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()  # keep BN running stats fixed too

    def forward(self, x, return_attention: bool = False):
        mid_feat, final_feat = self.backbone(x)  # (B, C_mid, 14,14), (B, C_final, 7,7) for 224px

        mid_feat = self.cbam_mid(mid_feat)        # locally attention-refined, mid stage
        final_feat = self.cbam_final(final_feat)  # locally attention-refined, final stage

        # Global context built jointly from BOTH original (pre-fusion) CBAM
        # maps' tokens -- unchanged from v3.
        global_ctx = self.transformer([mid_feat, final_feat])  # (B, embed_dim)
        gate = self.gate_mlp(global_ctx)  # (B, C_final) context-conditioned gate, sigmoid in [0, 1]

        # FPN-style lateral fusion: downsample+project mid-stage detail to
        # the final stage's shape and add it in, scaled by a zero-init
        # learnable scalar (see __init__ comment) so training starts
        # identical to "final_feat only" and gradually blends in real
        # mid-stage spatial content as `lateral_scale` grows from 0.
        lateral = self.lateral_bn(self.lateral_conv(mid_feat))  # (B, C_final, 7,7)
        fused_feat = final_feat + self.lateral_scale * lateral

        # Additive/residual gating instead of pure multiplicative gating
        # (see v2 notes): at initialization gate ~= 0.5 uniformly, so
        # `feat * (1 + gate)` starts close to an identity-like feedback path
        # (`feat * 1.5`) rather than immediately suppressing the pretrained
        # backbone's features by ~50% before anything has been learned.
        # The gate is now applied to the FUSED map (v4), so the global
        # context reweights channels of a map that actually carries
        # mid-stage spatial detail, not just final-stage semantics.
        gated_feat = fused_feat * (1 + gate.unsqueeze(-1).unsqueeze(-1))

        pooled = self.pool(gated_feat).flatten(1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, gated_feat, gate
        return logits


class FrozenBackboneMinimalNet(nn.Module):
    """Round 6 sanity-check floor: frozen ResNet50 backbone (identical
    features_only extraction as ContextGatedCBATNet, final stage only) ->
    global-average-pool -> dropout(0.3) -> linear classifier. No CBAM, no
    fusion, no Transformer branch, no gating -- the simplest possible model
    on top of the same pretrained/frozen features, used to isolate whether
    CG-CBAT's extra modules add ANY value over just the frozen backbone's
    features with a fresh linear head.

    Reuses ContextGatedCBATNet's own checkpoint-loading/freezing machinery
    by construction (same backbone_name, same MID/FINAL stride resolution),
    just via a much shorter forward pass using only the final stage map.
    """

    MID_STRIDE = ContextGatedCBATNet.MID_STRIDE
    FINAL_STRIDE = ContextGatedCBATNet.FINAL_STRIDE

    def __init__(self, num_classes: int, img_size: int = 224,
                 backbone_name: str = "resnet50"):
        super().__init__()
        probe = timm.create_model(backbone_name, pretrained=False, features_only=True)
        reductions = probe.feature_info.reduction()
        channels_all = probe.feature_info.channels()
        del probe

        mid_idx = reductions.index(self.MID_STRIDE)
        final_idx = reductions.index(self.FINAL_STRIDE)
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True,
            out_indices=(mid_idx, final_idx),
        )
        final_channels = channels_all[final_idx]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(final_channels, num_classes)

    # Same checkpoint-loading / freezing behaviour as ContextGatedCBATNet;
    # duplicated (not inherited) to keep this class trivially simple and
    # independently readable as the "floor" reference.
    load_backbone_checkpoint = ContextGatedCBATNet.load_backbone_checkpoint
    freeze_backbone = ContextGatedCBATNet.freeze_backbone

    def forward(self, x):
        _mid_feat, final_feat = self.backbone(x)
        pooled = self.pool(final_feat).flatten(1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
