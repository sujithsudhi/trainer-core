"""Tests for trainer-core inference artifact export."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trainer_core import export_inference_artifact


class DummyPosition(nn.Module):
    def __init__(self,
                 max_len   : int,
                 embed_dim : int,
                ) -> None:
        super().__init__()
        self.max_len           = max_len
        self.positional_table  = nn.Parameter(torch.zeros(1, max_len, embed_dim))


class DummyDropPath(nn.Module):
    def __init__(self,
                 drop_prob : float,
                ) -> None:
        super().__init__()
        self.drop_prob = drop_prob


class DummyAttention(nn.Module):
    def __init__(self,
                 embed_dim         : int,
                 num_heads         : int,
                 attention_dropout : float,
                 qkv_bias          : bool,
                ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout_p = attention_dropout
        self.w_q       = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
        self.w_k       = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
        self.w_v       = nn.Linear(embed_dim, embed_dim, bias = qkv_bias)
        self.w_o       = nn.Linear(embed_dim, embed_dim)
        self.rope      = None


class DummyFeedForward(nn.Module):
    def __init__(self,
                 embed_dim  : int,
                 hidden_dim : int,
                 dropout    : float,
                ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(embed_dim, hidden_dim)
        self.fc2     = nn.Linear(hidden_dim, embed_dim)


class DummyResidual(nn.Module):
    def __init__(self,
                 embed_dim      : int,
                 module         : nn.Module,
                 dropout        : float,
                 layer_norm_eps : float,
                 drop_path      : float,
                ) -> None:
        super().__init__()
        self.norm_first = True
        self.module     = module
        self.dropout    = nn.Dropout(dropout)
        self.drop_path  = DummyDropPath(drop_path)
        self.norm       = nn.LayerNorm(embed_dim, eps = layer_norm_eps)


class DummyTextLayer(nn.Module):
    def __init__(self,
                 embed_dim         : int,
                 num_heads         : int,
                 mlp_hidden_dim    : int,
                 dropout           : float,
                 attention_dropout : float,
                 qkv_bias          : bool,
                 layer_norm_eps    : float,
                 drop_path         : float,
                ) -> None:
        super().__init__()
        self.residual_attention = DummyResidual(embed_dim      = embed_dim,
                                                module         = DummyAttention(embed_dim,
                                                                                num_heads,
                                                                                attention_dropout,
                                                                                qkv_bias),
                                                dropout        = dropout,
                                                layer_norm_eps = layer_norm_eps,
                                                drop_path      = drop_path)
        self.residual_mlp       = DummyResidual(embed_dim      = embed_dim,
                                                module         = DummyFeedForward(embed_dim,
                                                                                  mlp_hidden_dim,
                                                                                  dropout),
                                                dropout        = dropout,
                                                layer_norm_eps = layer_norm_eps,
                                                drop_path      = drop_path)


class DummyTextModel(nn.Module):
    def __init__(self,
                ) -> None:
        super().__init__()
        self.use_rope      = False
        self.use_cls_token = True
        self.pooling       = "cls"

        self.token_embedding = nn.Embedding(30522, 128, padding_idx = 0)
        self.position        = DummyPosition(max_len = 512, embed_dim = 128)
        self.encoder         = nn.ModuleList([DummyTextLayer(embed_dim         = 128,
                                                             num_heads         = 4,
                                                             mlp_hidden_dim    = 256,
                                                             dropout           = 0.1,
                                                             attention_dropout = 0.1,
                                                             qkv_bias          = True,
                                                             layer_norm_eps    = 1e-5,
                                                             drop_path         = 0.0)
                                              for _ in range(4)])
        self.norm            = nn.LayerNorm(128, eps = 1e-5)
        self.head            = nn.Sequential(nn.Linear(128, 128),
                                             nn.GELU(),
                                             nn.Dropout(0.1),
                                             nn.Linear(128, 1))
        self.cls_token       = nn.Parameter(torch.zeros(1, 1, 128))


class DummyPatchEmbedding(nn.Module):
    def __init__(self,
                 image_size  : int,
                 patch_size  : int,
                 in_channels : int,
                 embed_dim   : int,
                ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.proj       = nn.Conv2d(in_channels,
                                    embed_dim,
                                    kernel_size = patch_size,
                                    stride      = patch_size)


class DummyBlockConfig:
    def __init__(self,
                 attention_type : str,
                 rope_base      : int,
                 window_size    : int | None,
                ) -> None:
        self.attention_type = attention_type
        self.rope_base      = rope_base
        self.window_size    = window_size

    @property
    def is_local(self) -> bool:
        return self.attention_type == "local"


class DummyBackbone(nn.Module):
    def __init__(self,
                ) -> None:
        super().__init__()
        self.embed_dim     = 64
        self.use_cls_token = True
        self.patch_embed   = DummyPatchEmbedding(image_size = 224,
                                                 patch_size = 16,
                                                 in_channels = 3,
                                                 embed_dim = 64)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, 64))
        self.pos_embed     = nn.Parameter(torch.zeros(1, 197, 64))
        self.blocks        = nn.ModuleList([DummyTextLayer(embed_dim         = 64,
                                                           num_heads         = 4,
                                                           mlp_hidden_dim    = 128,
                                                           dropout           = 0.1,
                                                           attention_dropout = 0.1,
                                                           qkv_bias          = True,
                                                           layer_norm_eps    = 1e-6,
                                                           drop_path         = 0.0)
                                            for _ in range(4)])
        self.norm          = nn.LayerNorm(64, eps = 1e-6)
        self.block_configs = [DummyBlockConfig("local", 10_000, 7),
                              DummyBlockConfig("local", 10_000, 7),
                              DummyBlockConfig("local", 10_000, 7),
                              DummyBlockConfig("global", 1_000_000, None)]


class DummyDetectionHead(nn.Module):
    def __init__(self,
                ) -> None:
        super().__init__()
        self.query_embed       = nn.Parameter(torch.zeros(1, 20, 64))
        self.query_norm        = nn.LayerNorm(64, eps = 1e-5)
        self.memory_norm       = nn.LayerNorm(64, eps = 1e-5)
        self.cross_attention   = nn.MultiheadAttention(embed_dim   = 64,
                                                       num_heads   = 4,
                                                       dropout     = 0.1,
                                                       batch_first = True)
        self.attention_dropout = nn.Dropout(0.1)
        self.ffn_norm          = nn.LayerNorm(64, eps = 1e-5)
        self.ffn               = nn.Sequential(nn.Linear(64, 128),
                                               nn.GELU(),
                                               nn.Dropout(0.1),
                                               nn.Linear(128, 64))
        self.ffn_dropout       = nn.Dropout(0.1)
        self.box_head          = nn.Sequential(nn.LayerNorm(64, eps = 1e-5),
                                               nn.Linear(64, 128),
                                               nn.GELU(),
                                               nn.Linear(128, 4))
        self.objectness_head   = nn.Sequential(nn.LayerNorm(64, eps = 1e-5),
                                               nn.Linear(64, 1))
        self.class_head        = nn.Sequential(nn.LayerNorm(64, eps = 1e-5),
                                               nn.Linear(64, 10))


class DummyVisionModel(nn.Module):
    def __init__(self,
                ) -> None:
        super().__init__()
        self.backbone       = DummyBackbone()
        self.detection_head = DummyDetectionHead()

    @property
    def block_configs(self) -> list[DummyBlockConfig]:
        return self.backbone.block_configs


class DummyTokenizer:
    def save(self,
             path : str,
            ) -> None:
        Path(path).write_text("{\"type\":\"dummy-tokenizer\"}\n", encoding = "utf-8")


class DummyTokenizerWrapper:
    def __init__(self,
                 tokenizer : DummyTokenizer,
                ) -> None:
        self.tokenizer = tokenizer


def _read_safetensors_header(path : Path,
                            ) -> dict[str, dict[str, object]]:
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), byteorder = "little", signed = False)
        header_bytes  = handle.read(header_length)
    return json.loads(header_bytes.decode("utf-8"))


class ExportingTests(unittest.TestCase):
    def test_export_text_artifact_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model      = DummyTextModel()
            tokenizer  = DummyTokenizerWrapper(DummyTokenizer())
            export_dir = Path(temp_dir) / "imdb-bert-base-uncased"

            export_inference_artifact(model, tokenizer, export_dir)

            manifest = json.loads((export_dir / "artifact.json").read_text(encoding = "utf-8"))
            model_js = json.loads((export_dir / "model.json").read_text(encoding = "utf-8"))
            weights  = _read_safetensors_header(export_dir / "weights.safetensors")

            self.assertEqual(manifest["artifact_name"], "imdb-bert-base-uncased")
            self.assertEqual(manifest["model_family"], "encoder")
            self.assertEqual(manifest["task"], "text")
            self.assertEqual(manifest["weight_format"], "safetensors")
            self.assertEqual(manifest["files"]["tokenizer"], "tokenizer/tokenizer.json")

            self.assertEqual(model_js["format"], "safetensors")
            self.assertEqual(model_js["config"]["name"], "encoder_classifier")
            self.assertEqual(model_js["config"]["model"]["vocab_size"], 30522)
            self.assertEqual(model_js["config"]["model"]["max_length"], 512)
            self.assertEqual(model_js["config"]["model"]["embed_dim"], 128)
            self.assertEqual(model_js["config"]["model"]["depth"], 4)
            self.assertEqual(model_js["config"]["model"]["num_heads"], 4)
            self.assertEqual(model_js["config"]["model"]["mlp_hidden_dim"], 256)
            self.assertEqual(model_js["config"]["model"]["cls_head_dim"], 128)
            self.assertEqual(model_js["config"]["model"]["num_outputs"], 1)
            self.assertFalse(model_js["config"]["model"]["use_rope"])

            graph_ops = [node["op"] for node in model_js["builder"]["graph"]["nodes"]]
            self.assertEqual(graph_ops,
                             ["token_embedding",
                              "positional_encoding",
                              "transformer_encoder",
                              "layer_norm",
                              "classifier_head"])

            self.assertEqual(set(weights.keys()), set(model.state_dict().keys()))
            self.assertEqual(weights["token_embedding.weight"]["dtype"], "F32")
            self.assertEqual((export_dir / "tokenizer" / "tokenizer.json").read_text(encoding = "utf-8"),
                             "{\"type\":\"dummy-tokenizer\"}\n")

    def test_export_vision_artifact_bundle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model      = DummyVisionModel()
            export_dir = Path(temp_dir) / "vision-detector"

            export_inference_artifact(model, None, export_dir)

            manifest = json.loads((export_dir / "artifact.json").read_text(encoding = "utf-8"))
            model_js = json.loads((export_dir / "model.json").read_text(encoding = "utf-8"))
            weights  = _read_safetensors_header(export_dir / "weights.safetensors")

            self.assertEqual(manifest["artifact_name"], "vision-detector")
            self.assertEqual(manifest["model_family"], "vision-language")
            self.assertEqual(manifest["task"], "vision")
            self.assertEqual(manifest["weight_format"], "safetensors")
            self.assertNotIn("tokenizer", manifest["files"])

            self.assertEqual(model_js["config"]["name"], "vision_detector")
            self.assertEqual(model_js["config"]["model"]["backbone"]["image_size"], 224)
            self.assertEqual(model_js["config"]["model"]["backbone"]["patch_size"], 16)
            self.assertEqual(model_js["config"]["model"]["backbone"]["num_layers"], 4)
            self.assertEqual(model_js["config"]["model"]["backbone"]["block_pattern"],
                             ["local", "local", "local", "global"])
            self.assertEqual(model_js["config"]["model"]["head"]["num_queries"], 20)
            self.assertEqual(model_js["config"]["model"]["head"]["num_classes"], 10)

            graph_ops = [node["op"] for node in model_js["builder"]["graph"]["nodes"]]
            self.assertEqual(graph_ops, ["patch_embedding", "vision_backbone", "detection_head"])

            self.assertIn("backbone.patch_embed.proj.weight", weights)
            self.assertIn("detection_head.query_embed", weights)


if __name__ == "__main__":
    unittest.main()
