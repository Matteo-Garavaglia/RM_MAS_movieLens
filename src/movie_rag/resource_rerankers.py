from __future__ import annotations

import gc
from typing import Sequence

import numpy as np


class SentenceTransformerCrossEncoderScorer:
    """Lazy CrossEncoder scorer exposing ``score(query, documents)``.

    The scorer deliberately exposes the small interface already consumed by
    ``ScoreRerankerAdapter``.  Precision is selected conservatively:
    ``auto`` uses FP16 on CUDA and leaves the framework default on MPS/CPU.
    """

    def __init__(
        self,
        model_name: str,
        *,
        name: str | None = None,
        trust_remote_code: bool = False,
        batch_size: int = 8,
        max_length: int = 1024,
        device: str = "auto",
        precision: str = "auto",
    ) -> None:
        self.model_name = str(model_name)
        self.name = name or self.model_name
        self.trust_remote_code = bool(trust_remote_code)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.device = str(device)
        self.precision = str(precision)
        self.model = None
        self._resolved_device: str | None = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self):
        if self.model is not None:
            return self

        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install sentence-transformers and torch to use a CrossEncoder reranker"
            ) from exc

        device = self._resolve_device()
        model_kwargs = {}
        if self.precision == "float16":
            model_kwargs["dtype"] = torch.float16
        elif self.precision == "float32":
            model_kwargs["dtype"] = torch.float32
        elif self.precision == "auto":
            # CUDA benefits strongly from FP16 for these rerankers.  Keep the
            # framework default on MPS/CPU for stability and portability.
            if device.startswith("cuda"):
                model_kwargs["dtype"] = torch.float16
        else:
            raise ValueError(
                "precision must be one of: auto, float16, float32"
            )

        self.model = CrossEncoder(
            self.model_name,
            device=device,
            trust_remote_code=self.trust_remote_code,
            max_length=self.max_length,
            model_kwargs=model_kwargs or None,
        )
        self._resolved_device = device
        return self

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        self.load()
        docs = [str(document) for document in documents]
        if not docs:
            return np.empty(0, dtype=np.float64)
        pairs = [(str(query), document) for document in docs]
        values = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(scores) != len(docs):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(docs)} documents"
            )
        return scores

    def unload(self) -> None:
        self.model = None
        gc.collect()
        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass


__all__ = ["SentenceTransformerCrossEncoderScorer"]
