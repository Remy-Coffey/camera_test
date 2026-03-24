from __future__ import annotations

import base64
import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

from domain import ActivitySegment


T = TypeVar("T")


def _chunked(items: list[T], size: int) -> Iterable[list[T]]:
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def _extract_json_object(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = _strip_code_fence(raw_text)
    if not text:
        return None, None
    try:
        parsed = json.loads(text)
        return (parsed if isinstance(parsed, dict) else None), "json"
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None, None
    try:
        parsed = json.loads(match.group(0))
        return (parsed if isinstance(parsed, dict) else None), "embedded_json"
    except json.JSONDecodeError:
        return None, None


def _extract_weak_description(raw_text: str) -> str | None:
    text = _strip_code_fence(raw_text)
    text = re.sub(r"^\s*(description|描述|summary|摘要)\s*[:：]?\s*", "", text, flags=re.I)
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = " ".join(lines[:3]).strip()
    return candidate[:240] if candidate else None


def _is_low_quality_description(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return True
    generic_phrases = (
        "有人",
        "人物停留",
        "有人停留",
        "人物移动",
        "有人移动",
        "停留",
        "移动",
        "画面中有1人",
        "one person",
        "person is present",
    )
    return any(phrase in normalized for phrase in generic_phrases) and len(normalized) <= 18


class LLMProvider:
    provider_name = "noop"

    def health(self, configured_model: str | None = None) -> dict[str, Any]:
        return {
            "enabled": False,
            "provider": self.provider_name,
            "base_url": None,
            "configured_model": configured_model,
            "reachable": False,
            "model_installed": False,
            "installed_models": [],
        }


@dataclass(slots=True)
class OllamaProviderBase(LLMProvider):
    base_url: str
    model: str
    timeout_seconds: int
    max_retries: int

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        started = time.perf_counter()
        for _ in range(max(1, self.max_retries)):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    return {
                        "ok": True,
                        "data": json.loads(response.read().decode("utf-8")),
                        "status": "success",
                        "error": None,
                        "latency_ms": latency_ms,
                    }
            except socket.timeout as exc:
                last_error = exc
            except urllib.error.URLError as exc:
                last_error = exc
            except json.JSONDecodeError as exc:
                last_error = exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        if isinstance(last_error, socket.timeout):
            status = "timeout"
        elif isinstance(last_error, json.JSONDecodeError):
            status = "invalid_json"
        else:
            status = "request_failed"
        return {
            "ok": False,
            "data": None,
            "status": status,
            "error": str(last_error) if last_error else "request_failed",
            "latency_ms": latency_ms,
        }

    def health(self, configured_model: str | None = None) -> dict[str, Any]:
        reachable = False
        model_installed = False
        installed_models: list[str] = []
        target_model = configured_model or self.model
        try:
            with urllib.request.urlopen(f"{self.base_url.rstrip('/')}/api/tags", timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            reachable = True
            installed_models = [str(item.get("name", "")).strip() for item in data.get("models", [])]
            model_installed = target_model in installed_models
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass

        return {
            "enabled": True,
            "provider": self.provider_name,
            "base_url": self.base_url,
            "configured_model": target_model,
            "reachable": reachable,
            "model_installed": model_installed,
            "installed_models": installed_models,
        }


@dataclass(slots=True)
class OllamaTextProvider(OllamaProviderBase):
    provider_name = "ollama_text"

    def enhance_segments_debug(self, segments: list[ActivitySegment], batch_size: int, model: str | None = None) -> list[dict[str, Any] | None]:
        target_model = model or self.model
        status = self.health(target_model)
        if not status["reachable"] or not status["model_installed"]:
            return [
                {
                    "status": "not_called",
                    "fallback_reason": "text_model_unavailable",
                    "latency_ms": 0,
                    "output": None,
                    "debug": {
                        "text_model": target_model,
                        "text_provider": self.provider_name,
                        "text_status": "not_called",
                        "text_fallback_reason": "text_model_unavailable",
                    },
                }
                for _ in segments
            ]

        outputs: list[dict[str, Any] | None] = []
        for batch in _chunked(segments, batch_size):
            outputs.extend(self._enhance_batch_debug(batch, target_model))
        return outputs

    def _enhance_batch_debug(self, batch: list[ActivitySegment], model: str) -> list[dict[str, Any] | None]:
        prompt = self._build_batch_prompt(batch)
        response = self._post_json("/api/generate", {"model": model, "prompt": prompt, "stream": False, "format": "json"})
        raw_response = str((response.get("data") or {}).get("response", "")) if response["ok"] else ""
        parsed, parse_mode = _extract_json_object(raw_response) if raw_response else (None, None)
        items = parsed.get("segments", []) if parsed else []
        by_index = {
            int(item.get("index")): str(item.get("description", "")).strip() or None
            for item in items
            if str(item.get("index", "")).isdigit()
        }

        outputs: list[dict[str, Any]] = []
        for index in range(len(batch)):
            description = by_index.get(index)
            status = "success" if description else ("empty_response" if response["ok"] and not raw_response else response["status"])
            outputs.append(
                {
                    "status": status,
                    "fallback_reason": None if description else status,
                    "latency_ms": response["latency_ms"],
                    "output": {"description": description} if description else None,
                    "debug": {
                        "text_model": model,
                        "text_provider": self.provider_name,
                        "text_status": status,
                        "text_fallback_reason": None if description else status,
                        "text_prompt": prompt,
                        "text_raw_response": raw_response,
                        "text_parse_ok": bool(parsed),
                        "text_parse_mode": parse_mode,
                        "text_latency_ms": response["latency_ms"],
                    },
                }
            )
        return outputs

    def _build_batch_prompt(self, batch: list[ActivitySegment]) -> str:
        records = []
        for index, segment in enumerate(batch):
            records.append(
                {
                    "index": index,
                    "rule_description": segment.rule_description,
                    "video_description": segment.video_description,
                    "action": segment.action,
                    "scene": segment.scene,
                    "labels": segment.video_labels,
                    "features": segment.features,
                }
            )
        return (
            "你是监控视频分析助手。请把每个片段的事实整理成克制、清楚、自然的中文摘要。\n"
            "要求：\n"
            "1. 不要编造画面中没有出现的人数、动作或风险。\n"
            "2. 如果视频模型已经给出更具体的行为描述，优先保留视频层事实。\n"
            "3. 返回 JSON，格式为 {\"segments\":[{\"index\":0,\"description\":\"...\"}]}\n"
            f"segments={json.dumps(records, ensure_ascii=False)}"
        )


@dataclass(slots=True)
class OllamaVisionProvider(OllamaProviderBase):
    provider_name = "ollama_vision"

    def analyze_segments_debug(
        self,
        segments: list[ActivitySegment],
        segment_images: list[list[Path]],
        batch_size: int,
        model: str | None = None,
    ) -> list[dict[str, Any] | None]:
        del batch_size
        target_model = model or self.model
        status = self.health(target_model)
        if not status["reachable"] or not status["model_installed"]:
            return [
                {
                    "output": None,
                    "status": "not_called",
                    "fallback_reason": "video_model_unavailable",
                    "video_result_status": "fallback",
                    "parse_mode": None,
                    "raw_response_present": False,
                    "latency_ms": 0,
                    "debug": {
                        "video_model": target_model,
                        "video_provider": self.provider_name,
                        "vision_status": "not_called",
                        "vision_fallback_reason": "video_model_unavailable",
                        "vision_latency_ms": 0,
                    },
                }
                for _ in segments
            ]

        outputs: list[dict[str, Any] | None] = []
        for segment, images in zip(segments, segment_images):
            outputs.append(self._analyze_single_segment_debug(segment, images, target_model))
        return outputs

    def _analyze_single_segment_debug(self, segment: ActivitySegment, images: list[Path], model: str) -> dict[str, Any] | None:
        if not images:
            return {
                "output": None,
                "status": "not_called",
                "fallback_reason": "missing_keyframes",
                "video_result_status": "fallback",
                "parse_mode": None,
                "raw_response_present": False,
                "latency_ms": 0,
                "debug": {
                    "video_model": model,
                    "video_provider": self.provider_name,
                    "vision_status": "not_called",
                    "vision_fallback_reason": "missing_keyframes",
                    "vision_latency_ms": 0,
                },
            }

        encoded_images = [base64.b64encode(image.read_bytes()).decode("utf-8") for image in images]
        prompt = (
            "请分析这段监控视频的关键帧。你会看到同一片段的多张图像。\n"
            "目标是描述画面里真正可见的事实：人数变化、人物状态、动作、互动和场景变化。\n"
            "要求：\n"
            "1. 优先描述可见动作，例如站立等待、交谈、走动、围观、鼓掌、举手、跑动、坐着、看向某处。\n"
            "2. 如果看不出具体动作，也要明确说是站立、坐着、等待或视线变化，不能只写“有人停留”。\n"
            "3. 只有在画面支持时，才能参考规则层信息。\n"
            "4. 返回 JSON，格式为 "
            "{\"description\":\"...\",\"labels\":[\"...\"],\"action\":\"...\",\"scene\":\"...\",\"confidence\":0.0,\"evidence_frames\":[0,1]}\n"
            f"规则层信息：{json.dumps(segment.features, ensure_ascii=False)}"
        )

        response = self._post_json(
            "/api/chat",
            {
                "model": model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": "你是监控视频多模态分析助手，必须基于可见画面回答，不能编造。"},
                    {"role": "user", "content": prompt, "images": encoded_images},
                ],
            },
        )
        raw_response = str(((response.get("data") or {}).get("message") or {}).get("content", "")).strip() if response["ok"] else ""
        parsed, parse_mode = _extract_json_object(raw_response) if raw_response else (None, None)

        if parsed:
            description = str(parsed.get("description", "")).strip()
            if description and not _is_low_quality_description(description):
                return {
                    "output": {
                        "description": description,
                        "labels": [str(label).strip() for label in parsed.get("labels", []) if str(label).strip()],
                        "action": str(parsed.get("action", "")).strip() or None,
                        "scene": str(parsed.get("scene", "")).strip() or None,
                        "confidence": float(parsed["confidence"]) if parsed.get("confidence") is not None else None,
                        "evidence_frames": [int(item) for item in parsed.get("evidence_frames", []) if str(item).isdigit()],
                    },
                    "status": "success",
                    "fallback_reason": None,
                    "video_result_status": "success",
                    "parse_mode": parse_mode,
                    "raw_response_present": bool(raw_response),
                    "latency_ms": response["latency_ms"],
                    "debug": {
                        "video_model": model,
                        "video_provider": self.provider_name,
                        "vision_status": "success",
                        "vision_fallback_reason": None,
                        "vision_prompt": prompt,
                        "vision_raw_response": raw_response,
                        "vision_parse_ok": True,
                        "vision_parse_mode": parse_mode,
                        "vision_latency_ms": response["latency_ms"],
                    },
                }
            if description:
                return {
                    "output": None,
                    "status": "low_quality_description",
                    "fallback_reason": "low_quality_description",
                    "video_result_status": "fallback",
                    "parse_mode": parse_mode,
                    "raw_response_present": bool(raw_response),
                    "latency_ms": response["latency_ms"],
                    "debug": {
                        "video_model": model,
                        "video_provider": self.provider_name,
                        "vision_status": "low_quality_description",
                        "vision_fallback_reason": "low_quality_description",
                        "vision_prompt": prompt,
                        "vision_raw_response": raw_response,
                        "vision_parse_ok": True,
                        "vision_parse_mode": parse_mode,
                        "vision_latency_ms": response["latency_ms"],
                    },
                }

        weak_description = _extract_weak_description(raw_response)
        if weak_description and not _is_low_quality_description(weak_description):
            return {
                "output": {
                    "description": weak_description,
                    "labels": [],
                    "action": None,
                    "scene": None,
                    "confidence": None,
                    "evidence_frames": [],
                },
                "status": "weak_text_only",
                "fallback_reason": None,
                "video_result_status": "weak_success",
                "parse_mode": "weak_text",
                "raw_response_present": bool(raw_response),
                "latency_ms": response["latency_ms"],
                "debug": {
                    "video_model": model,
                    "video_provider": self.provider_name,
                    "vision_status": "weak_text_only",
                    "vision_fallback_reason": None,
                    "vision_prompt": prompt,
                    "vision_raw_response": raw_response,
                    "vision_parse_ok": False,
                    "vision_parse_mode": "weak_text",
                    "vision_latency_ms": response["latency_ms"],
                },
            }

        if not response["ok"]:
            status = response["status"]
        elif not raw_response:
            status = "empty_response"
        else:
            status = "invalid_json"
        return {
            "output": None,
            "status": status,
            "fallback_reason": status,
            "video_result_status": "fallback",
            "parse_mode": parse_mode,
            "raw_response_present": bool(raw_response),
            "latency_ms": response["latency_ms"],
            "debug": {
                "video_model": model,
                "video_provider": self.provider_name,
                "vision_status": status,
                "vision_fallback_reason": status,
                "vision_prompt": prompt,
                "vision_raw_response": raw_response,
                "vision_parse_ok": False,
                "vision_parse_mode": parse_mode,
                "vision_latency_ms": response["latency_ms"],
            },
        }


class NoopProvider(LLMProvider):
    pass


class DescriptionEnhancer:
    def __init__(self, provider: OllamaTextProvider | NoopProvider, batch_size: int):
        self.provider = provider
        self.batch_size = batch_size

    def enhance(self, segments: list[ActivitySegment], model: str | None = None) -> list[dict[str, Any] | None]:
        if isinstance(self.provider, NoopProvider):
            return [None for _ in segments]
        return self.provider.enhance_segments_debug(segments, self.batch_size, model=model)

    def health(self, model: str | None = None) -> dict[str, Any]:
        return self.provider.health(model)


class VideoEnhancer:
    def __init__(self, provider: OllamaVisionProvider | NoopProvider, batch_size: int):
        self.provider = provider
        self.batch_size = batch_size

    def analyze(self, segments: list[ActivitySegment], segment_images: list[list[Path]], model: str | None = None) -> list[dict[str, Any] | None]:
        if isinstance(self.provider, NoopProvider):
            return [None for _ in segments]
        return self.provider.analyze_segments_debug(segments, segment_images, self.batch_size, model=model)

    def health(self, model: str | None = None) -> dict[str, Any]:
        return self.provider.health(model)
