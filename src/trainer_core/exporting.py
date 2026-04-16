"""Inference artifact export helpers for trainer-core models."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn


_SAFETENSORS_DTYPES: dict[torch.dtype, str] = {
    torch.bool     : "BOOL",
    torch.uint8    : "U8",
    torch.int8     : "I8",
    torch.int16    : "I16",
    torch.int32    : "I32",
    torch.int64    : "I64",
    torch.float16  : "F16",
    torch.bfloat16 : "BF16",
    torch.float32  : "F32",
    torch.float64  : "F64",
}


def _coalesce(*values : Any,
             ) -> Any:
    """
    Return the first non-None value from a candidate list.
    Args:
        *values : Candidate values evaluated in order.
    Returns:
        The first value that is not `None`, otherwise `None`.
    """
    for value in values:
        if value is not None:
            return value
    return None


def _count_indexed_prefix(state_dict : dict[str, torch.Tensor],
                          prefix     : str,
                         ) -> int:
    """
    Count distinct numeric indices under one dotted state-dict prefix.
    Args:
        state_dict : Mapping of parameter names to tensors.
        prefix     : Prefix such as `encoder.` or `backbone.blocks.`.
    Returns:
        Number of distinct numeric indices discovered under the prefix.
    """
    indices: set[int] = set()

    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue

        index_str = key[len(prefix):].split(".", maxsplit = 1)[0]
        if index_str.isdigit():
            indices.add(int(index_str))

    return len(indices)


def _normalize_task(task : str,
                   ) -> str:
    """
    Normalize one user-facing task hint into a supported internal task name.
    Args:
        task : User-facing task value or alias.
    Returns:
        Normalized task name.
    Raises:
        ValueError: Raised when the task hint is unsupported.
    """
    normalized = task.strip().lower().replace("-", "_")

    if normalized in {"text", "encoder", "encoder_classifier", "imdb", "sentiment"}:
        return "text"

    if normalized in {"vision", "vision_detector", "detector", "object_detection"}:
        return "vision"

    raise ValueError(f"Unsupported export task '{task}'. Expected text or vision.")


def _dropout_probability(module : Any,
                        ) -> Optional[float]:
    """
    Extract a dropout probability from one module-like object when available.
    Args:
        module : Candidate module exposing a `p` attribute.
    Returns:
        Dropout probability, or `None` when unavailable.
    """
    if module is None:
        return None
    probability = getattr(module, "p", None)
    if probability is None:
        return None
    return float(probability)


def _artifact_name(model      : nn.Module,
                   output_dir : Path,
                  ) -> str:
    """
    Resolve the exported artifact name.
    Args:
        model      : Model being exported.
        output_dir : Destination artifact directory.
    Returns:
        Stable artifact name written into `artifact.json`.
    """
    config = getattr(model, "config", None)
    return str(_coalesce(getattr(model, "artifact_name", None),
                         getattr(config, "name", None),
                         output_dir.name))


def _extract_state_dict(model : nn.Module,
                       ) -> dict[str, torch.Tensor]:
    """
    Extract one CPU state dict from the provided model.
    Args:
        model : PyTorch model whose parameters should be exported.
    Returns:
        Mapping of PyTorch-style parameter names to CPU tensors.
    Raises:
        ValueError: Raised when the model does not expose any tensors.
    """
    state_dict: dict[str, torch.Tensor] = {}

    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        state_dict[key] = value.detach().cpu().contiguous()

    if not state_dict:
        raise ValueError("Model state_dict is empty; unable to export an artifact.")

    return state_dict


def _infer_task(model      : nn.Module,
                state_dict : dict[str, torch.Tensor],
                task       : Optional[str],
               ) -> str:
    """
    Resolve the export task from an explicit hint or model structure.
    Args:
        model      : Model being exported.
        state_dict : Extracted model state dict.
        task       : Optional explicit task hint.
    Returns:
        Normalized internal task name.
    Raises:
        ValueError: Raised when the task cannot be inferred.
    """
    if task is not None:
        return _normalize_task(task)

    if hasattr(model, "detection_head"):
        return "vision"

    if hasattr(model, "token_embedding") or "token_embedding.weight" in state_dict:
        return "text"

    if "detection_head.query_embed" in state_dict:
        return "vision"

    raise ValueError("Unable to infer export task from the provided model. Pass task='text' or task='vision'.")


def _classifier_hidden_dim(head : Any,
                          ) -> Optional[int]:
    """
    Resolve the optional hidden dimension used by a classifier head.
    Args:
        head : Classifier head module.
    Returns:
        Hidden dimension for two-layer classifier heads, otherwise `None`.
    """
    if isinstance(head, nn.Sequential):
        for module in head:
            if isinstance(module, nn.Linear):
                return int(module.out_features)
    return None


def _classifier_output_dim(head       : Any,
                           state_dict : dict[str, torch.Tensor],
                          ) -> int:
    """
    Resolve the number of classifier outputs from a head module or state dict.
    Args:
        head       : Classifier head module.
        state_dict : Exported model state dict.
    Returns:
        Number of outputs produced by the classifier.
    Raises:
        ValueError: Raised when the output dimension cannot be determined.
    """
    if isinstance(head, nn.Linear):
        return int(head.out_features)

    if isinstance(head, nn.Sequential):
        linear_layers = [module for module in head if isinstance(module, nn.Linear)]
        if linear_layers:
            return int(linear_layers[-1].out_features)

    head_weight = state_dict.get("head.weight")
    if head_weight is not None:
        return int(head_weight.shape[0])

    final_weight = state_dict.get("head.3.weight")
    if final_weight is not None:
        return int(final_weight.shape[0])

    raise ValueError("Unable to resolve classifier output dimension for export.")


def _resolve_text_model_config(model      : nn.Module,
                               state_dict : dict[str, torch.Tensor],
                              ) -> dict[str, Any]:
    """
    Resolve one encoder-classifier config from the live model structure.
    Args:
        model      : Text model compatible with the encoder-classifier runtime.
        state_dict : Exported model state dict.
    Returns:
        Serializable resolved text-model config.
    Raises:
        ValueError: Raised when required model structure is missing.
    """
    token_embedding = getattr(model, "token_embedding", None)
    position        = getattr(model, "position", None)
    encoder         = getattr(model, "encoder", None)
    norm            = getattr(model, "norm", None)
    head            = getattr(model, "head", None)
    config          = getattr(model, "config", None)

    token_weight = state_dict.get("token_embedding.weight")
    if token_embedding is None and token_weight is None:
        raise ValueError("Text export requires a token_embedding module or token_embedding.weight tensor.")

    if encoder is None or len(encoder) == 0:
        depth = _count_indexed_prefix(state_dict, "encoder.")
        if depth <= 0:
            raise ValueError("Text export requires encoder layers under the encoder.* prefix.")
        raise ValueError("Text export requires the live model to expose encoder layers for config resolution.")

    first_layer      = encoder[0]
    attention_block  = getattr(first_layer, "residual_attention", None)
    mlp_block        = getattr(first_layer, "residual_mlp", None)
    attention_module = getattr(attention_block, "module", None)
    mlp_module       = getattr(mlp_block, "module", None)

    if attention_module is None or mlp_module is None:
        raise ValueError("Text export requires encoder layers with residual_attention and residual_mlp modules.")

    position_tensor = state_dict.get("position.positional_table")
    hidden_weight   = state_dict.get("encoder.0.residual_mlp.module.fc1.weight")
    q_bias          = state_dict.get("encoder.0.residual_attention.module.w_q.bias")

    if token_embedding is not None:
        vocab_size = int(token_embedding.num_embeddings)
        embed_dim  = int(token_embedding.embedding_dim)
    else:
        vocab_size = int(token_weight.shape[0])
        embed_dim  = int(token_weight.shape[1])

    mlp_hidden_dim = int(_coalesce(getattr(getattr(mlp_module, "fc1", None), "out_features", None),
                                   hidden_weight.shape[0] if hidden_weight is not None else None,
                                   getattr(config, "mlp_hidden_dim", None),
                                  ))
    classifier_hidden_dim = _classifier_hidden_dim(head)
    if classifier_hidden_dim is None:
        head0_weight = state_dict.get("head.0.weight")
        if head0_weight is not None:
            classifier_hidden_dim = int(head0_weight.shape[0])

    return {"vocab_size"         : vocab_size,
            "max_length"         : int(_coalesce(getattr(position, "max_len", None),
                                                 position_tensor.shape[1] if position_tensor is not None and position_tensor.ndim > 1 else None,
                                                 getattr(config, "max_length", None),
                                                 512,
                                                )),
            "embed_dim"          : embed_dim,
            "depth"              : len(encoder),
            "num_heads"          : int(_coalesce(getattr(attention_module, "num_heads", None),
                                                 getattr(config, "num_heads", None),
                                                )),
            "mlp_ratio"          : float(_coalesce(getattr(config, "mlp_ratio", None),
                                                   mlp_hidden_dim / max(embed_dim, 1),
                                                  )),
            "mlp_hidden_dim"     : mlp_hidden_dim,
            "dropout"            : float(_coalesce(getattr(config, "dropout", None),
                                                   _dropout_probability(getattr(mlp_module, "dropout", None)),
                                                   _dropout_probability(getattr(attention_block, "dropout", None)),
                                                   0.0,
                                                  )),
            "attention_dropout"  : float(_coalesce(getattr(config, "attention_dropout", None),
                                                   getattr(attention_module, "dropout_p", None),
                                                   _dropout_probability(getattr(attention_module, "dropout", None)),
                                                   0.0,
                                                  )),
            "qkv_bias"           : bool(_coalesce(getattr(getattr(attention_module, "w_q", None), "bias", None) is not None,
                                                  q_bias is not None,
                                                  getattr(config, "qkv_bias", None),
                                                  True,
                                                 )),
            "pre_norm"           : bool(_coalesce(getattr(attention_block, "norm_first", None),
                                                  getattr(config, "pre_norm", None),
                                                  True,
                                                 )),
            "layer_norm_eps"     : float(_coalesce(getattr(norm, "eps", None),
                                                   getattr(getattr(attention_block, "norm", None), "eps", None),
                                                   getattr(config, "layer_norm_eps", None),
                                                   1e-5,
                                                  )),
            "drop_path"          : float(_coalesce(getattr(getattr(attention_block, "drop_path", None), "drop_prob", None),
                                                   getattr(config, "drop_path", None),
                                                   0.0,
                                                  )),
            "use_cls_token"      : bool(_coalesce(getattr(model, "use_cls_token", None),
                                                  "cls_token" in state_dict,
                                                  False,
                                                 )),
            "cls_head_dim"       : classifier_hidden_dim,
            "num_outputs"        : _classifier_output_dim(head, state_dict),
            "pooling"            : str(_coalesce(getattr(model, "pooling", None),
                                                 getattr(config, "pooling", None),
                                                 "cls",
                                                )),
            "use_rope"           : bool(_coalesce(getattr(model, "use_rope", None),
                                                  getattr(attention_module, "rope", None) is not None,
                                                  position is None,
                                                 )),
            "rope_base"          : int(_coalesce(getattr(config, "rope_base", None), 10000))}


def _resolve_block_pattern(block_configs : Any,
                          ) -> list[str]:
    """
    Resolve the ordered local/global block pattern for one vision backbone.
    Args:
        block_configs : Sequence of block configs exposed by the backbone.
    Returns:
        Ordered block pattern for export.
    """
    if block_configs is None:
        return []
    return ["local" if bool(getattr(block_config, "is_local", False)) else "global"
            for block_config in block_configs]


def _resolve_vision_model_config(model      : nn.Module,
                                 state_dict : dict[str, torch.Tensor],
                                ) -> dict[str, Any]:
    """
    Resolve one vision-detector config from the live model structure.
    Args:
        model      : Vision detector compatible with the runtime graph builder.
        state_dict : Exported model state dict.
    Returns:
        Serializable resolved vision-detector config.
    Raises:
        ValueError: Raised when required model structure is missing.
    """
    backbone       = getattr(model, "backbone", None)
    detection_head = getattr(model, "detection_head", None)

    if backbone is None or detection_head is None:
        raise ValueError("Vision export requires model.backbone and model.detection_head.")

    patch_embed   = getattr(backbone, "patch_embed", None)
    blocks        = getattr(backbone, "blocks", None)
    block_configs = _coalesce(getattr(model, "block_configs", None),
                              getattr(backbone, "block_configs", None))
    norm          = getattr(backbone, "norm", None)

    if patch_embed is None or blocks is None or len(blocks) == 0 or norm is None:
        raise ValueError("Vision export requires a backbone with patch_embed, blocks, and norm modules.")

    first_block      = blocks[0]
    attention_block  = getattr(first_block, "residual_attention", None)
    mlp_block        = getattr(first_block, "residual_mlp", None)
    attention_module = getattr(attention_block, "module", None)
    mlp_module       = getattr(mlp_block, "module", None)

    if attention_module is None or mlp_module is None:
        raise ValueError("Vision export requires transformer-style backbone blocks.")

    patch_projection = getattr(patch_embed, "proj", None)
    if patch_projection is None:
        raise ValueError("Vision export requires patch_embed.proj for patch embedding weights.")

    block_pattern    = _resolve_block_pattern(block_configs)
    local_block_cfg  = next((cfg for cfg in block_configs if bool(getattr(cfg, "is_local", False))), None) if block_configs else None
    global_block_cfg = next((cfg for cfg in block_configs if not bool(getattr(cfg, "is_local", False))), None) if block_configs else None
    query_embed      = getattr(detection_head, "query_embed", None)
    cross_attention  = getattr(detection_head, "cross_attention", None)
    class_head       = getattr(detection_head, "class_head", None)
    box_head         = getattr(detection_head, "box_head", None)
    ffn              = getattr(detection_head, "ffn", None)
    query_norm       = getattr(detection_head, "query_norm", None)
    query_weight     = state_dict.get("detection_head.query_embed")
    class_weight     = state_dict.get("detection_head.class_head.1.weight")
    ffn_weight       = state_dict.get("detection_head.ffn.0.weight")
    box_weight       = state_dict.get("detection_head.box_head.1.weight")

    if query_embed is None and query_weight is None:
        raise ValueError("Vision export requires detection_head.query_embed.")

    if cross_attention is None:
        raise ValueError("Vision export requires detection_head.cross_attention.")

    if class_head is None and class_weight is None:
        raise ValueError("Vision export requires detection_head.class_head.")

    head_linear = None
    if isinstance(class_head, nn.Sequential):
        for module in class_head:
            if isinstance(module, nn.Linear):
                head_linear = module

    ffn_fc1 = ffn[0] if isinstance(ffn, nn.Sequential) and len(ffn) > 0 else None
    box_fc1 = box_head[1] if isinstance(box_head, nn.Sequential) and len(box_head) > 1 else None

    mlp_hidden_dim = int(_coalesce(getattr(getattr(mlp_module, "fc1", None), "out_features", None),
                                   state_dict["backbone.blocks.0.residual_mlp.module.fc1.weight"].shape[0]
                                   if "backbone.blocks.0.residual_mlp.module.fc1.weight" in state_dict
                                   else None,
                                  ))
    embed_dim = int(_coalesce(getattr(backbone, "embed_dim", None),
                              getattr(patch_projection, "out_channels", None),
                             ))

    backbone_cfg = {"image_size"        : int(getattr(patch_embed, "image_size", 0)),
                    "patch_size"        : int(getattr(patch_embed, "patch_size", 0)),
                    "in_channels"       : int(getattr(patch_projection, "in_channels", 0)),
                    "embed_dim"         : embed_dim,
                    "num_layers"        : len(blocks),
                    "num_heads"         : int(_coalesce(getattr(attention_module, "num_heads", None), 0)),
                    "mlp_ratio"         : float(mlp_hidden_dim / max(embed_dim, 1)),
                    "mlp_hidden_dim"    : mlp_hidden_dim,
                    "attention_dropout" : float(_coalesce(getattr(attention_module, "dropout_p", None),
                                                          _dropout_probability(getattr(attention_module, "dropout", None)),
                                                          0.0,
                                                         )),
                    "dropout"           : float(_coalesce(_dropout_probability(getattr(mlp_module, "dropout", None)),
                                                          _dropout_probability(getattr(attention_block, "dropout", None)),
                                                          0.0,
                                                         )),
                    "qkv_bias"          : bool(getattr(getattr(attention_module, "w_q", None), "bias", None) is not None),
                    "use_cls_token"     : bool(_coalesce(getattr(backbone, "use_cls_token", None),
                                                          "backbone.cls_token" in state_dict,
                                                          False,
                                                         )),
                    "use_rope"          : bool(getattr(attention_module, "rope", None) is not None
                                               or local_block_cfg is not None
                                               or global_block_cfg is not None),
                    "layer_norm_eps"    : float(getattr(norm, "eps", 1e-6)),
                    "local_window_size" : int(_coalesce(getattr(local_block_cfg, "window_size", None), 7)),
                    "local_rope_base"   : int(_coalesce(getattr(local_block_cfg, "rope_base", None), 10_000)),
                    "global_rope_base"  : int(_coalesce(getattr(global_block_cfg, "rope_base", None), 1_000_000)),
                    "block_pattern"     : block_pattern or ["global"] * len(blocks)}

    head_cfg = {"num_queries"    : int(_coalesce(query_embed.shape[1] if query_embed is not None else None,
                                                 query_weight.shape[1] if query_weight is not None else None,
                                                )),
                "num_classes"    : int(_coalesce(getattr(head_linear, "out_features", None),
                                                 class_weight.shape[0] if class_weight is not None else None,
                                                )),
                "num_heads"      : int(getattr(cross_attention, "num_heads", 0)),
                "mlp_hidden_dim" : int(_coalesce(getattr(ffn_fc1, "out_features", None),
                                                 ffn_weight.shape[0] if ffn_weight is not None else None,
                                                 getattr(box_fc1, "out_features", None),
                                                 box_weight.shape[0] if box_weight is not None else None,
                                                 embed_dim,
                                                )),
                "dropout"        : float(_coalesce(_dropout_probability(getattr(detection_head, "attention_dropout", None)),
                                                   _dropout_probability(getattr(detection_head, "ffn_dropout", None)),
                                                   0.0,
                                                  )),
                "layer_norm_eps" : float(_coalesce(getattr(query_norm, "eps", None), 1e-5))}

    return {"backbone" : backbone_cfg,
            "head"     : head_cfg}


def _text_graph(model_config : dict[str, Any],
               ) -> dict[str, Any]:
    """
    Build graph metadata for the encoder-classifier runtime.
    Args:
        model_config : Resolved text-model config.
    Returns:
        Serializable graph payload for `model.json`.
    """
    nodes = [{"name"         : "token_embedding",
              "op"           : "token_embedding",
              "inputs"       : ["input_ids"],
              "outputs"      : ["embedded_tokens"],
              "param_prefix" : "token_embedding.",
              "attrs"        : {"vocab_size" : model_config["vocab_size"],
                                "embed_dim"  : model_config["embed_dim"]}}]

    if not model_config["use_rope"]:
        nodes.append({"name"         : "position",
                      "op"           : "positional_encoding",
                      "inputs"       : ["embedded_tokens"],
                      "outputs"      : ["positioned_tokens"],
                      "param_prefix" : "position.",
                      "attrs"        : {"max_length" : model_config["max_length"],
                                        "use_rope"   : False}})

    encoder_input = "positioned_tokens" if not model_config["use_rope"] else "embedded_tokens"
    nodes.append({"name"         : "encoder",
                  "op"           : "transformer_encoder",
                  "inputs"       : [encoder_input, "attention_mask"],
                  "outputs"      : ["encoded_tokens"],
                  "param_prefix" : "encoder.",
                  "attrs"        : {"depth"             : model_config["depth"],
                                    "num_heads"         : model_config["num_heads"],
                                    "mlp_ratio"         : model_config["mlp_ratio"],
                                    "mlp_hidden_dim"    : model_config["mlp_hidden_dim"],
                                    "dropout"           : model_config["dropout"],
                                    "attention_dropout" : model_config["attention_dropout"],
                                    "qkv_bias"          : model_config["qkv_bias"],
                                    "pre_norm"          : model_config["pre_norm"],
                                    "layer_norm_eps"    : model_config["layer_norm_eps"],
                                    "drop_path"         : model_config["drop_path"],
                                    "pooling"           : model_config["pooling"],
                                    "use_rope"          : model_config["use_rope"],
                                    "rope_base"         : model_config["rope_base"]}})
    nodes.append({"name"         : "norm",
                  "op"           : "layer_norm",
                  "inputs"       : ["encoded_tokens"],
                  "outputs"      : ["normalized_tokens"],
                  "param_prefix" : "norm."})
    nodes.append({"name"         : "classifier",
                  "op"           : "classifier_head",
                  "inputs"       : ["normalized_tokens"],
                  "outputs"      : ["logits"],
                  "param_prefix" : "head.",
                  "attrs"        : {"cls_head_dim" : model_config["cls_head_dim"],
                                    "num_outputs"  : model_config["num_outputs"]}})

    return {"version" : "inference.graph/1",
            "inputs"  : ["input_ids", "attention_mask"],
            "outputs" : ["logits"],
            "nodes"   : nodes}


def _vision_graph(model_config : dict[str, Any],
                 ) -> dict[str, Any]:
    """
    Build graph metadata for the vision-detector runtime.
    Args:
        model_config : Resolved vision-detector config.
    Returns:
        Serializable graph payload for `model.json`.
    """
    backbone_cfg = model_config["backbone"]
    head_cfg     = model_config["head"]

    return {"version" : "inference.graph/1",
            "inputs"  : ["image"],
            "outputs" : ["pred_boxes", "pred_objectness_logits", "pred_class_logits"],
            "nodes"   : [{"name"         : "patch_embedding",
                          "op"           : "patch_embedding",
                          "inputs"       : ["image"],
                          "outputs"      : ["patch_tokens"],
                          "param_prefix" : "backbone.patch_embed.",
                          "attrs"        : {"image_size"  : backbone_cfg["image_size"],
                                            "patch_size"  : backbone_cfg["patch_size"],
                                            "in_channels" : backbone_cfg["in_channels"],
                                            "embed_dim"   : backbone_cfg["embed_dim"]}},
                         {"name"         : "vision_backbone",
                          "op"           : "vision_backbone",
                          "inputs"       : ["patch_tokens"],
                          "outputs"      : ["vision_features"],
                          "param_prefix" : "backbone.",
                          "attrs"        : {"num_layers"        : backbone_cfg["num_layers"],
                                            "num_heads"         : backbone_cfg["num_heads"],
                                            "mlp_ratio"         : backbone_cfg["mlp_ratio"],
                                            "mlp_hidden_dim"    : backbone_cfg["mlp_hidden_dim"],
                                            "attention_dropout" : backbone_cfg["attention_dropout"],
                                            "dropout"           : backbone_cfg["dropout"],
                                            "qkv_bias"          : backbone_cfg["qkv_bias"],
                                            "use_cls_token"     : backbone_cfg["use_cls_token"],
                                            "use_rope"          : backbone_cfg["use_rope"],
                                            "layer_norm_eps"    : backbone_cfg["layer_norm_eps"],
                                            "local_window_size" : backbone_cfg["local_window_size"],
                                            "local_rope_base"   : backbone_cfg["local_rope_base"],
                                            "global_rope_base"  : backbone_cfg["global_rope_base"],
                                            "block_pattern"     : backbone_cfg["block_pattern"]}},
                         {"name"         : "detection_head",
                          "op"           : "detection_head",
                          "inputs"       : ["vision_features"],
                          "outputs"      : ["pred_boxes", "pred_objectness_logits", "pred_class_logits"],
                          "param_prefix" : "detection_head.",
                          "attrs"        : {"num_queries"    : head_cfg["num_queries"],
                                            "num_classes"    : head_cfg["num_classes"],
                                            "num_heads"      : head_cfg["num_heads"],
                                            "mlp_hidden_dim" : head_cfg["mlp_hidden_dim"],
                                            "dropout"        : head_cfg["dropout"],
                                            "layer_norm_eps" : head_cfg["layer_norm_eps"]}}]}


def _tensor_data_bytes(tensor : torch.Tensor,
                      ) -> bytes:
    """
    Convert one CPU tensor into its raw byte payload for safetensors export.
    Args:
        tensor : Tensor to serialize.
    Returns:
        Raw byte payload for the tensor contents.
    """
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _write_safetensors(path       : Path,
                       state_dict : dict[str, torch.Tensor],
                      ) -> Path:
    """
    Write one minimal safetensors file for the provided state dict.
    Args:
        path       : Target safetensors path.
        state_dict : Mapping of tensor names to CPU tensors.
    Returns:
        Resolved safetensors file path.
    Raises:
        ValueError: Raised when a tensor dtype is unsupported.
    """
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents = True, exist_ok = True)

    header: dict[str, Any] = {}
    offset = 0

    for key, tensor in state_dict.items():
        dtype_name = _SAFETENSORS_DTYPES.get(tensor.dtype)
        if dtype_name is None:
            raise ValueError(f"Unsupported tensor dtype '{tensor.dtype}' for safetensors export.")

        data = _tensor_data_bytes(tensor)
        header[key] = {"dtype"        : dtype_name,
                       "shape"        : list(tensor.shape),
                       "data_offsets" : [offset, offset + len(data)]}
        offset += len(data)

    header_bytes = json.dumps(header, separators = (",", ":")).encode("utf-8")
    padding      = (8 - (len(header_bytes) % 8)) % 8
    if padding:
        header_bytes += b" " * padding

    with resolved.open("wb") as handle:
        handle.write(len(header_bytes).to_bytes(8, byteorder = "little", signed = False))
        handle.write(header_bytes)
        for tensor in state_dict.values():
            handle.write(_tensor_data_bytes(tensor))

    return resolved


def _copy_tokenizer_json(tokenizer  : Any,
                         output_dir : Path,
                        ) -> Optional[str]:
    """
    Copy or serialize the training tokenizer into the export bundle.
    Args:
        tokenizer  : Tokenizer object, wrapper, file path, or directory.
        output_dir : Artifact directory receiving tokenizer assets.
    Returns:
        Manifest-relative tokenizer path, or `None` when no tokenizer is supplied.
    Raises:
        ValueError: Raised when the tokenizer cannot be serialized to tokenizer.json.
    """
    if tokenizer is None:
        return None

    if hasattr(tokenizer, "tokenizer") and getattr(tokenizer, "tokenizer") is not tokenizer:
        return _copy_tokenizer_json(tokenizer.tokenizer, output_dir)

    tokenizer_dir  = output_dir / "tokenizer"
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    tokenizer_dir.mkdir(parents = True, exist_ok = True)

    if isinstance(tokenizer, (str, Path)):
        source = Path(tokenizer).expanduser().resolve()
        if source.is_dir():
            source = source / "tokenizer.json"
        if not source.exists():
            raise ValueError(f"Tokenizer source not found: {source}")
        if source.suffix.lower() != ".json":
            raise ValueError(f"Tokenizer export expects tokenizer.json, got '{source.name}'.")
        shutil.copy2(source, tokenizer_path)
        return tokenizer_path.relative_to(output_dir).as_posix()

    if hasattr(tokenizer, "save_pretrained"):
        with tempfile.TemporaryDirectory() as temp_dir:
            tokenizer.save_pretrained(temp_dir)
            source = Path(temp_dir) / "tokenizer.json"
            if source.exists():
                shutil.copy2(source, tokenizer_path)
                return tokenizer_path.relative_to(output_dir).as_posix()

    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None and hasattr(backend_tokenizer, "save"):
        backend_tokenizer.save(str(tokenizer_path))
        return tokenizer_path.relative_to(output_dir).as_posix()

    if hasattr(tokenizer, "save"):
        tokenizer.save(str(tokenizer_path))
        return tokenizer_path.relative_to(output_dir).as_posix()

    if hasattr(tokenizer, "to_str"):
        tokenizer_path.write_text(tokenizer.to_str(), encoding = "utf-8")
        return tokenizer_path.relative_to(output_dir).as_posix()

    if hasattr(tokenizer, "to_json"):
        payload = tokenizer.to_json()
        if isinstance(payload, str):
            tokenizer_path.write_text(payload, encoding = "utf-8")
        else:
            tokenizer_path.write_text(json.dumps(payload, indent = 2) + "\n", encoding = "utf-8")
        return tokenizer_path.relative_to(output_dir).as_posix()

    raise ValueError("Tokenizer must be a path, a tokenizer wrapper, or expose save/to_str/to_json methods.")


def export_inference_artifact(model      : nn.Module,
                              tokenizer  : Any,
                              output_dir : Path | str,
                              task       : Optional[str] = None,
                             ) -> Path:
    """
    Export one final inference artifact bundle for the provided model.
    Args:
        model      : Trained PyTorch model to export.
        tokenizer  : Real tokenizer used for training, or `None` for vision exports.
        output_dir : Destination directory for the artifact bundle.
        task       : Optional task hint. Supported values are `text` and `vision`.
    Returns:
        Resolved artifact directory path.
    Raises:
        ValueError: Raised when the model or tokenizer cannot be exported.
    """
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents = True, exist_ok = True)

    state_dict    = _extract_state_dict(model)
    resolved_task = _infer_task(model, state_dict, task)

    if resolved_task == "text":
        if tokenizer is None:
            raise ValueError("Text artifact export requires the tokenizer used for training.")
        model_name   = "encoder_classifier"
        model_family = "encoder"
        model_config = _resolve_text_model_config(model, state_dict)
        graph        = _text_graph(model_config)
    else:
        model_name   = "vision_detector"
        model_family = "vision-language"
        model_config = _resolve_vision_model_config(model, state_dict)
        graph        = _vision_graph(model_config)

    weights_path   = _write_safetensors(resolved_output_dir / "weights.safetensors", state_dict)
    tokenizer_path = _copy_tokenizer_json(tokenizer, resolved_output_dir)

    model_payload = {"builder" : {"model_type" : "graph",
                                  "graph"      : graph},
                     "config"  : {"name"  : model_name,
                                  "model" : model_config},
                     "format"  : "safetensors",
                     "source"  : {"framework" : "pytorch",
                                  "exporter"  : "trainer-core"}}

    files = {"metadata" : "model.json",
             "weights"  : weights_path.name}
    if tokenizer_path is not None:
        files["tokenizer"] = tokenizer_path

    manifest_payload = {"schema_version" : "inference.artifact/1",
                        "artifact_name"  : _artifact_name(model, resolved_output_dir),
                        "model_family"   : model_family,
                        "task"           : resolved_task,
                        "weight_format"  : "safetensors",
                        "files"          : files}

    (resolved_output_dir / "model.json").write_text(json.dumps(model_payload, indent = 2) + "\n",
                                                    encoding = "utf-8")
    (resolved_output_dir / "artifact.json").write_text(json.dumps(manifest_payload, indent = 2) + "\n",
                                                       encoding = "utf-8")
    return resolved_output_dir


__all__ = ["export_inference_artifact"]
