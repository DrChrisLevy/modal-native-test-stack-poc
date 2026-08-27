"""Lazy, offline-only access to the seven real Hugging Face models."""

from __future__ import annotations

import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lockfile import ModelManifest, ModelSpec, SnapshotResolver
from .outputs import (
    EntityCollection,
    ImageClassification,
    ImageEmbedding,
    LabelPrediction,
    NamedEntity,
    Sentiment,
    Summary,
    TextEmbedding,
    Transcription,
)


@dataclass(frozen=True, slots=True)
class ModelStatus:
    key: str
    repo_id: str
    revision: str
    task: str
    commit_pinned: bool
    available: bool
    loaded: bool


@dataclass(slots=True)
class _Bundle:
    model: Any
    processor: Any
    pipeline: Any | None = None


class ModelRegistry:
    """Loads each real model on first use from a mounted, pre-seeded snapshot.

    No Hugging Face import happens when this module or registry is constructed.
    More importantly, loaders receive a filesystem path rather than a repo ID and
    always set ``local_files_only=True``. An empty Volume therefore fails clearly
    instead of silently downloading weights during a test run.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        models_root: str | Path = "/models",
        *,
        device: str = "cpu",
        require_commit_pins: bool = False,
    ) -> None:
        if require_commit_pins:
            manifest.require_commit_pins()
        self.manifest = manifest
        self.resolver = SnapshotResolver(models_root)
        self.device = device
        self._bundles: dict[str, _Bundle] = {}
        self._load_lock = threading.RLock()
        self._inference_locks = {spec.key: threading.Lock() for spec in self.manifest.models}

    @classmethod
    def from_lockfile(
        cls,
        lockfile_path: str | Path,
        models_root: str | Path = "/models",
        *,
        device: str = "cpu",
        require_commit_pins: bool = False,
    ) -> ModelRegistry:
        return cls(
            ModelManifest.load(lockfile_path),
            models_root,
            device=device,
            require_commit_pins=require_commit_pins,
        )

    @property
    def specs(self) -> tuple[ModelSpec, ...]:
        return self.manifest.models

    @property
    def loaded_keys(self) -> frozenset[str]:
        return frozenset(self._bundles)

    def get_spec(self, key: str) -> ModelSpec:
        try:
            return self.manifest.by_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown model capability: {key!r}") from exc

    def snapshot_path(self, key: str) -> Path:
        return self.resolver.resolve(self.get_spec(key))

    def status(self) -> tuple[ModelStatus, ...]:
        loaded = self.loaded_keys
        return tuple(
            ModelStatus(
                key=spec.key,
                repo_id=spec.repo_id,
                revision=spec.revision,
                task=spec.task,
                commit_pinned=spec.is_commit_pinned,
                available=self.resolver.is_available(spec),
                loaded=spec.key in loaded,
            )
            for spec in self.specs
        )

    def all_snapshots_available(self) -> bool:
        return all(self.resolver.is_available(spec) for spec in self.specs)

    def load(self, key: str) -> None:
        """Load one capability without running inference."""

        self._bundle(key)

    def unload(self, key: str | None = None) -> None:
        """Drop cached model references, mainly for bounded-memory test shards."""

        with self._load_lock:
            if key is None:
                self._bundles.clear()
            else:
                self._bundles.pop(key, None)

    def embed_text(self, text: str) -> TextEmbedding:
        cleaned = _required_text(text)
        bundle = self._bundle("text_embedding")
        with self._inference_locks["text_embedding"]:
            vector = bundle.model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return TextEmbedding(vector=tuple(float(value) for value in vector.tolist()))

    def sentiment(self, text: str) -> Sentiment:
        cleaned = _required_text(text)
        bundle = self._bundle("sentiment")
        import torch

        with self._inference_locks["sentiment"], torch.inference_mode():
            encoded = bundle.processor(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            logits = bundle.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            index = int(probabilities.argmax().item())
            score = float(probabilities[index].item())
        labels = bundle.model.config.id2label
        label = str(labels.get(index, labels.get(str(index), str(index))))
        return Sentiment(label=label, score=score)

    def named_entities(self, text: str) -> EntityCollection:
        cleaned = _required_text(text)
        bundle = self._bundle("named_entities")
        with self._inference_locks["named_entities"]:
            raw_entities = bundle.pipeline(cleaned)
        entities = tuple(
            NamedEntity(
                text=str(entity.get("word", "")).strip(),
                label=str(entity.get("entity_group", entity.get("entity", "UNKNOWN"))),
                score=float(entity["score"]),
                start=_optional_int(entity.get("start")),
                end=_optional_int(entity.get("end")),
            )
            for entity in raw_entities
        )
        return EntityCollection(entities=entities)

    def summarize(
        self,
        text: str,
        *,
        max_new_tokens: int = 64,
        num_beams: int = 2,
    ) -> Summary:
        cleaned = _required_text(text)
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        if not 1 <= num_beams <= 8:
            raise ValueError("num_beams must be between 1 and 8")
        bundle = self._bundle("summary")
        import torch

        prompt = f"summarize: {cleaned}"
        with self._inference_locks["summary"], torch.inference_mode():
            encoded = bundle.processor(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            generated = bundle.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=num_beams > 1,
            )
            summary = bundle.processor.decode(generated[0], skip_special_tokens=True).strip()
        return Summary(text=summary)

    def classify_image(
        self,
        image: bytes | bytearray | memoryview | str | Path | Any,
        *,
        top_k: int = 5,
    ) -> ImageClassification:
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        prepared = _prepare_image(image)
        bundle = self._bundle("image_classification")
        import torch

        with self._inference_locks["image_classification"], torch.inference_mode():
            encoded = bundle.processor(images=prepared, return_tensors="pt")
            logits = bundle.model(**encoded).logits[0]
            probabilities = torch.softmax(logits, dim=-1)
            count = min(top_k, int(probabilities.shape[-1]))
            scores, indices = probabilities.topk(count)
        labels = bundle.model.config.id2label
        predictions = tuple(
            LabelPrediction(
                label=str(labels.get(int(index), labels.get(str(int(index)), int(index)))),
                score=float(score),
            )
            for score, index in zip(scores.tolist(), indices.tolist(), strict=True)
        )
        return ImageClassification(predictions=predictions)

    def embed_image(
        self, image: bytes | bytearray | memoryview | str | Path | Any
    ) -> ImageEmbedding:
        prepared = _prepare_image(image)
        bundle = self._bundle("image_embedding")
        import torch

        with self._inference_locks["image_embedding"], torch.inference_mode():
            encoded = bundle.processor(images=prepared, return_tensors="pt")
            features = bundle.model.get_image_features(**encoded)[0]
            features = features / features.norm(p=2).clamp_min(1e-12)
        return ImageEmbedding(vector=tuple(float(value) for value in features.tolist()))

    def transcribe(
        self,
        audio: bytes | bytearray | memoryview | str | Path | tuple[Any, int] | Any,
        *,
        sample_rate: int | None = None,
        max_new_tokens: int = 128,
    ) -> Transcription:
        samples, source_rate = _prepare_audio(audio, sample_rate=sample_rate)
        target_rate = 16_000
        if source_rate != target_rate:
            import librosa

            samples = librosa.resample(
                samples,
                orig_sr=source_rate,
                target_sr=target_rate,
                res_type="soxr_hq",
            )
        duration = float(len(samples) / target_rate)
        bundle = self._bundle("speech_to_text")
        import torch

        with self._inference_locks["speech_to_text"], torch.inference_mode():
            encoded = bundle.processor(
                samples,
                sampling_rate=target_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            generated = bundle.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            text = bundle.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return Transcription(text=text, duration_seconds=duration, sample_rate=target_rate)

    def _bundle(self, key: str) -> _Bundle:
        bundle = self._bundles.get(key)
        if bundle is not None:
            return bundle
        with self._load_lock:
            bundle = self._bundles.get(key)
            if bundle is not None:
                return bundle
            spec = self.get_spec(key)
            path = self.resolver.resolve(spec)
            bundle = self._load_bundle(spec, path)
            self._bundles[key] = bundle
            return bundle

    def _load_bundle(self, spec: ModelSpec, path: Path) -> _Bundle:
        location = str(path)
        common = {
            "local_files_only": True,
            "trust_remote_code": False,
        }

        if spec.key == "text_embedding":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                location,
                device=self.device,
                local_files_only=True,
                trust_remote_code=False,
            )
            model.eval()
            return _Bundle(model=model, processor=None)

        if spec.key == "sentiment":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(location, **common)
            model = AutoModelForSequenceClassification.from_pretrained(location, **common)
            model.eval()
            return _Bundle(model=model, processor=tokenizer)

        if spec.key == "named_entities":
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(location, **common)
            model = AutoModelForTokenClassification.from_pretrained(location, **common)
            model.eval()
            inference_pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1 if self.device == "cpu" else self.device,
            )
            return _Bundle(model=model, processor=tokenizer, pipeline=inference_pipeline)

        if spec.key == "summary":
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(location, **common)
            model = AutoModelForSeq2SeqLM.from_pretrained(location, **common)
            model.eval()
            return _Bundle(model=model, processor=tokenizer)

        if spec.key == "image_classification":
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            processor = AutoImageProcessor.from_pretrained(location, **common)
            model = AutoModelForImageClassification.from_pretrained(location, **common)
            model.eval()
            return _Bundle(model=model, processor=processor)

        if spec.key == "image_embedding":
            from transformers import AutoProcessor, CLIPModel

            processor = AutoProcessor.from_pretrained(location, **common)
            model = CLIPModel.from_pretrained(location, **common)
            model.eval()
            return _Bundle(model=model, processor=processor)

        if spec.key == "speech_to_text":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            processor = AutoProcessor.from_pretrained(location, **common)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(location, **common)
            model.eval()
            return _Bundle(model=model, processor=processor)

        raise KeyError(f"no loader is registered for {spec.key!r}")


def _required_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text must not be empty")
    return cleaned


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _prepare_image(image: bytes | bytearray | memoryview | str | Path | Any) -> Any:
    from PIL import Image, ImageOps

    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as opened:
            prepared = ImageOps.exif_transpose(opened).convert("RGB").copy()
    elif isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            prepared = ImageOps.exif_transpose(opened).convert("RGB").copy()
    elif isinstance(image, Image.Image):
        prepared = ImageOps.exif_transpose(image).convert("RGB")
    else:
        raise TypeError("image must be encoded bytes, a path, or a PIL Image")
    if prepared.width < 1 or prepared.height < 1:
        raise ValueError("image must contain at least one pixel")
    return prepared


def _prepare_audio(
    audio: bytes | bytearray | memoryview | str | Path | tuple[Any, int] | Any,
    *,
    sample_rate: int | None,
) -> tuple[Any, int]:
    import numpy as np
    import soundfile as sf

    if isinstance(audio, tuple) and len(audio) == 2:
        samples, tuple_rate = audio
        resolved_rate = int(tuple_rate)
        array = np.asarray(samples, dtype=np.float32)
    elif isinstance(audio, (bytes, bytearray, memoryview)):
        array, resolved_rate = sf.read(io.BytesIO(bytes(audio)), dtype="float32", always_2d=False)
    elif isinstance(audio, (str, Path)):
        array, resolved_rate = sf.read(str(audio), dtype="float32", always_2d=False)
    else:
        if sample_rate is None:
            raise TypeError("array audio requires an explicit sample_rate")
        array = np.asarray(audio, dtype=np.float32)
        resolved_rate = sample_rate

    if sample_rate is not None and not isinstance(audio, tuple):
        # Encoded files carry their own authoritative rate; reject accidental mismatch.
        if isinstance(audio, (bytes, bytearray, memoryview, str, Path)):
            if int(sample_rate) != int(resolved_rate):
                raise ValueError(
                    f"explicit sample_rate {sample_rate} does not match encoded audio "
                    f"rate {resolved_rate}"
                )
        else:
            resolved_rate = sample_rate
    resolved_rate = int(resolved_rate)
    if resolved_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if array.ndim == 2:
        array = array.mean(axis=1)
    elif array.ndim != 1:
        raise ValueError("audio must be mono samples or a frames-by-channels array")
    if array.size == 0:
        raise ValueError("audio must contain at least one sample")
    array = np.nan_to_num(array.astype(np.float32, copy=False))
    peak = float(np.max(np.abs(array)))
    if not math.isfinite(peak):
        raise ValueError("audio samples must be finite")
    if peak > 1.0:
        array = array / peak
    return array, resolved_rate
