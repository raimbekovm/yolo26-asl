"""ASL letter classifier models."""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.constants import ASL_CLASSES, IDX_TO_CLASS, NUM_KEYPOINTS


class ASLClassifier(nn.Module):
    """
    Base ASL classifier interface.

    Subclasses implement specific architectures (MLP, Transformer).
    """

    def __init__(self, num_classes: int = len(ASL_CLASSES)):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, keypoints: torch.Tensor) -> tuple[int, float, str]:
        """
        Predict ASL letter from keypoints.

        Args:
            keypoints: Tensor of shape (63,) or (21, 3).

        Returns:
            Tuple of (class_idx, confidence, class_name).
        """
        self.eval()
        with torch.no_grad():
            if keypoints.dim() == 1:
                keypoints = keypoints.unsqueeze(0)

            logits = self(keypoints)
            probs = F.softmax(logits, dim=-1)
            conf, idx = probs.max(dim=-1)

            idx = int(idx.item())
            conf = float(conf.item())
            name = IDX_TO_CLASS[idx]

            return idx, conf, name

    @classmethod
    def load(cls, path: Union[str, Path], device: str = "cpu") -> "ASLClassifier":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)

        # Determine model type from checkpoint
        if "model_type" in checkpoint:
            model_type = checkpoint["model_type"]
        else:
            # Infer from state dict
            model_type = "mlp"  # Default

        if model_type == "mlp":
            model = ASLClassifierMLP(**checkpoint.get("config", {}))
        elif model_type == "transformer":
            model = ASLClassifierTransformer(**checkpoint.get("config", {}))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()

        return model

    def save(self, path: Union[str, Path], config: Optional[dict] = None) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": self.state_dict(),
            "model_type": self.__class__.__name__.lower().replace("aslclassifier", ""),
            "config": config or {},
            "num_classes": self.num_classes,
        }
        torch.save(checkpoint, path)


class ASLClassifierMLP(ASLClassifier):
    """
    MLP-based ASL classifier.

    Simple but effective architecture for keypoint classification.

    Architecture:
        Input (63) -> FC(256) -> BN -> ReLU -> Dropout
                   -> FC(128) -> BN -> ReLU -> Dropout
                   -> FC(64)  -> BN -> ReLU -> Dropout
                   -> FC(31)  -> Output

    Example:
        >>> model = ASLClassifierMLP()
        >>> keypoints = torch.randn(32, 63)
        >>> logits = model(keypoints)
        >>> print(logits.shape)  # (32, 31)
    """

    def __init__(
        self,
        input_dim: int = NUM_KEYPOINTS * 3,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        num_classes: int = len(ASL_CLASSES),
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Initialize MLP classifier.

        Args:
            input_dim: Input feature dimension (default: 63 = 21*3).
            hidden_dims: Tuple of hidden layer dimensions.
            num_classes: Number of output classes.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__(num_classes)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm

        # Build layers
        layers = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 63) or (batch, 21, 3).

        Returns:
            Logits of shape (batch, num_classes).
        """
        if x.dim() == 3:
            x = x.flatten(1)

        features = self.features(x)
        logits = self.classifier(features)
        return logits

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
        }


class ASLClassifierTransformer(ASLClassifier):
    """
    Transformer-based ASL classifier.

    Treats each keypoint as a token and uses self-attention
    to model relationships between hand joints.

    Example:
        >>> model = ASLClassifierTransformer()
        >>> keypoints = torch.randn(32, 21, 3)
        >>> logits = model(keypoints)
        >>> print(logits.shape)  # (32, 31)
    """

    def __init__(
        self,
        num_keypoints: int = NUM_KEYPOINTS,
        keypoint_dim: int = 3,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = len(ASL_CLASSES),
        dropout: float = 0.1,
    ):
        """
        Initialize Transformer classifier.

        Args:
            num_keypoints: Number of keypoints (21 for hand).
            keypoint_dim: Dimension per keypoint (3 for x, y, conf).
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            mlp_ratio: MLP hidden dim ratio.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__(num_classes)

        self.num_keypoints = num_keypoints
        self.keypoint_dim = keypoint_dim
        self.embed_dim = embed_dim

        # Keypoint embedding
        self.keypoint_embed = nn.Linear(keypoint_dim, embed_dim)

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_keypoints + 1, embed_dim))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 63) or (batch, 21, 3).

        Returns:
            Logits of shape (batch, num_classes).
        """
        batch_size = x.shape[0]

        # Reshape if flattened
        if x.dim() == 2:
            x = x.view(batch_size, self.num_keypoints, self.keypoint_dim)

        # Embed keypoints
        x = self.keypoint_embed(x)  # (B, 21, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 22, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Extract CLS token
        cls_output = x[:, 0]

        # Classification
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)

        return logits

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "num_keypoints": self.num_keypoints,
            "keypoint_dim": self.keypoint_dim,
            "embed_dim": self.embed_dim,
            "num_classes": self.num_classes,
        }


def create_classifier(
    model_type: str = "mlp",
    **kwargs,
) -> ASLClassifier:
    """
    Factory function to create classifier.

    Args:
        model_type: Type of classifier ('mlp' or 'transformer').
        **kwargs: Additional arguments for the model.

    Returns:
        ASLClassifier instance.
    """
    if model_type == "mlp":
        return ASLClassifierMLP(**kwargs)
    elif model_type == "transformer":
        return ASLClassifierTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
