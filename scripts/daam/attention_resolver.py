from __future__ import annotations

import glob
import os
import re
from typing import Any, ClassVar

from modules.shared import opts

try:
    from dynamicprompts.commands import (
        SequenceCommand as DPSequenceCommand,
        VariantCommand as DPVariantCommand,
        WildcardCommand as DPWildcardCommand,
    )
    from dynamicprompts.commands.variable_commands import (
        VariableAccessCommand as DPVariableAccessCommand,
        VariableAssignmentCommand as DPVariableAssignmentCommand,
    )
    from dynamicprompts.commands.wrap_command import WrapCommand as DPWrapCommand
    from dynamicprompts.generators import CombinatorialPromptGenerator as DPCombinatorialPromptGenerator
    from dynamicprompts.parser.parse import ParserConfig as DPParserConfig
    from dynamicprompts.parser.parse import parse as dp_parse
    from dynamicprompts.wildcards import WildcardManager as DPWildcardManager
except Exception:
    DPSequenceCommand = None
    DPVariantCommand = None
    DPWildcardCommand = None
    DPVariableAccessCommand = None
    DPVariableAssignmentCommand = None
    DPWrapCommand = None
    DPCombinatorialPromptGenerator = None
    DPParserConfig = None
    DPWildcardManager = None
    dp_parse = None


class AttentionResolverMixin:
    DYNAMIC_RESOLVE_MAX_CANDIDATES: ClassVar[int] = 512
    DYNAMIC_RESOLVE_MAX_WILDCARD_VALUES: ClassVar[int] = 4096
    DYNAMIC_RESOLVE_CACHE_MAX_ENTRIES: ClassVar[int] = 4096
    _break_regex: ClassVar[re.Pattern[str]] = re.compile(r"\bBREAK\b", flags=re.IGNORECASE)
    _variant_block_regex: ClassVar[re.Pattern[str]] = re.compile(
        r"\{[^{}]*\|[^{}]*\}|\[[^\[\]]*\|[^\[\]]*\]"
    )
    _wildcard_token_regex: ClassVar[re.Pattern[str]] = re.compile(r"__([A-Za-z0-9_\-./\\*\[\]?]+)__")
    _extra_network_tag_regex: ClassVar[re.Pattern[str]] = re.compile(r"<[^<>:]+:[^<>]+>")
    _invalid_filename_chars: ClassVar[re.Pattern[str]] = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')
    _dp_generator_cache_key: ClassVar[Any | None] = None
    _dp_generator: ClassVar[Any | None] = None
    _dp_wildcard_manager: ClassVar[Any | None] = None
    _dp_resolve_cache: ClassVar[dict[str, list[str]]] = {}

    @classmethod
    def _split_attention_texts(cls, attention_texts: str):
        normalized = cls._break_regex.sub(",", attention_texts or "")
        return [s.strip() for s in normalized.split(",") if s.strip()]

    @classmethod
    def _normalize_for_match(cls, text: str):
        if not text:
            return ""
        normalized = text.lower()
        normalized = cls._break_regex.sub(" ", normalized)
        normalized = re.sub(r"<lora:[^>]+>", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @classmethod
    def _contains_phrase(cls, prompt: str, phrase: str):
        prompt_n = cls._normalize_for_match(prompt)
        phrase_n = cls._normalize_for_match(phrase)
        if not prompt_n or not phrase_n:
            return False
        return phrase_n in prompt_n

    @classmethod
    def _best_prompt_match(cls, prompt: str, candidates):
        best = None
        best_len = -1
        for candidate in candidates:
            candidate = (candidate or "").strip()
            if not candidate:
                continue
            if cls._contains_phrase(prompt, candidate):
                candidate_len = len(candidate)
                if candidate_len > best_len:
                    best = candidate
                    best_len = candidate_len
        return best

    @classmethod
    def _extract_variant_options(cls, token: str):
        token = (token or "").strip()
        if len(token) < 3:
            return []
        if not ((token.startswith("{") and token.endswith("}")) or (token.startswith("[") and token.endswith("]"))):
            return []
        inner = token[1:-1].strip()
        if "|" not in inner:
            return []
        # Dynamic prompts can prefix range/joiner blocks with '$$'.
        if "$$" in inner:
            inner = inner.rsplit("$$", 1)[-1]
        options = []
        for opt in inner.split("|"):
            cleaned = re.sub(r"^\s*[-+]?\d+(?:\.\d+)?::", "", opt).strip()
            if cleaned:
                options.append(cleaned)
        return options

    @staticmethod
    def _get_webui_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    @classmethod
    def _get_wildcard_dirs(cls):
        wildcard_dirs = []
        configured_wildcard_dir = getattr(opts, "wildcard_dir", None)
        if configured_wildcard_dir:
            wildcard_dirs.append(os.path.abspath(configured_wildcard_dir))
        wildcard_dirs.append(
            os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "sd-dynamic-prompts", "wildcards")
            )
        )
        wildcard_dirs.append(os.path.join(cls._get_webui_root(), "wildcards"))
        deduped = []
        for wildcard_dir in wildcard_dirs:
            if wildcard_dir and wildcard_dir not in deduped and os.path.isdir(wildcard_dir):
                deduped.append(wildcard_dir)
        return deduped

    @classmethod
    def _load_wildcard_values(cls, wildcard_name: str):
        wildcard_name = (wildcard_name or "").strip().replace("\\", "/")
        if not wildcard_name:
            return []
        has_glob = any(ch in wildcard_name for ch in ("*", "?", "["))
        values = []
        seen = set()
        for wildcard_dir in cls._get_wildcard_dirs():
            patterns = []
            if wildcard_name.endswith(".txt"):
                patterns.append(os.path.join(wildcard_dir, wildcard_name))
            else:
                patterns.append(os.path.join(wildcard_dir, wildcard_name + ".txt"))
            if has_glob:
                wildcard_pattern = wildcard_name if wildcard_name.endswith(".txt") else wildcard_name + ".txt"
                patterns.append(os.path.join(wildcard_dir, "**", wildcard_pattern))
            for pattern in patterns:
                for wildcard_file in glob.glob(pattern, recursive=has_glob):
                    if not os.path.isfile(wildcard_file):
                        continue
                    try:
                        with open(wildcard_file, "r", encoding="utf-8", errors="ignore") as f:
                            for line in f:
                                line = line.strip()
                                if not line or line.startswith("#"):
                                    continue
                                if line not in seen:
                                    seen.add(line)
                                    values.append(line)
                    except Exception:
                        continue
        return values

    @classmethod
    def _get_dynamicprompts_parser_config(cls):
        if DPParserConfig is None:
            return None
        try:
            variant_start = getattr(opts, "dp_parser_variant_start", "{") or "{"
            variant_end = getattr(opts, "dp_parser_variant_end", "}") or "}"
            wildcard_wrap = getattr(opts, "dp_parser_wildcard_wrap", "__") or "__"
            return DPParserConfig(
                variant_start=variant_start,
                variant_end=variant_end,
                wildcard_wrap=wildcard_wrap,
            )
        except Exception:
            return None

    @classmethod
    def _get_dynamicprompts_generator(cls):
        if DPCombinatorialPromptGenerator is None or DPWildcardManager is None:
            return None

        parser_config = cls._get_dynamicprompts_parser_config()
        wildcard_dirs = cls._get_wildcard_dirs()
        wildcard_root = wildcard_dirs[0] if len(wildcard_dirs) > 0 else None
        cache_key = (
            wildcard_root,
            getattr(parser_config, "variant_start", None),
            getattr(parser_config, "variant_end", None),
            getattr(parser_config, "wildcard_wrap", None),
            bool(getattr(opts, "dp_ignore_whitespace", False)),
        )

        if cls._dp_generator is not None and cls._dp_generator_cache_key == cache_key:
            return cls._dp_generator

        try:
            cls._dp_wildcard_manager = DPWildcardManager(wildcard_root) if wildcard_root else DPWildcardManager()
            kwargs = {
                "wildcard_manager": cls._dp_wildcard_manager,
                "ignore_whitespace": bool(getattr(opts, "dp_ignore_whitespace", False)),
            }
            if parser_config is not None:
                kwargs["parser_config"] = parser_config
            cls._dp_generator = DPCombinatorialPromptGenerator(**kwargs)
            cls._dp_generator_cache_key = cache_key
            cls._dp_resolve_cache = {}
            return cls._dp_generator
        except Exception:
            cls._dp_generator = None
            cls._dp_wildcard_manager = None
            cls._dp_generator_cache_key = None
            return None

    @classmethod
    def _collect_wildcard_names_from_command(cls, command, names: set, depth: int = 0):
        if command is None or depth > 12 or DPWildcardCommand is None:
            return

        if DPWildcardCommand is not None and isinstance(command, DPWildcardCommand):
            wildcard = getattr(command, "wildcard", "")
            if isinstance(wildcard, str):
                wildcard = wildcard.strip()
                if wildcard:
                    names.add(wildcard)
            else:
                cls._collect_wildcard_names_from_command(wildcard, names, depth + 1)
            wildcard_vars = getattr(command, "variables", None)
            if isinstance(wildcard_vars, dict):
                for value in wildcard_vars.values():
                    cls._collect_wildcard_names_from_command(value, names, depth + 1)
            return

        if DPSequenceCommand is not None and isinstance(command, DPSequenceCommand):
            for token in getattr(command, "tokens", []):
                cls._collect_wildcard_names_from_command(token, names, depth + 1)
            return

        if DPVariantCommand is not None and isinstance(command, DPVariantCommand):
            for option in getattr(command, "variants", []):
                cls._collect_wildcard_names_from_command(getattr(option, "value", None), names, depth + 1)
            return

        if DPVariableAssignmentCommand is not None and isinstance(command, DPVariableAssignmentCommand):
            cls._collect_wildcard_names_from_command(getattr(command, "value", None), names, depth + 1)
            return

        if DPVariableAccessCommand is not None and isinstance(command, DPVariableAccessCommand):
            cls._collect_wildcard_names_from_command(getattr(command, "default", None), names, depth + 1)
            return

        if DPWrapCommand is not None and isinstance(command, DPWrapCommand):
            cls._collect_wildcard_names_from_command(getattr(command, "wrapper", None), names, depth + 1)
            cls._collect_wildcard_names_from_command(getattr(command, "inner", None), names, depth + 1)
            return

    @classmethod
    def _dynamic_prompt_candidates(cls, text: str):
        text = (text or "").strip()
        if not text:
            return []

        cached = cls._dp_resolve_cache.get(text)
        if isinstance(cached, list):
            return list(cached)

        candidates = []
        generator = cls._get_dynamicprompts_generator()
        wildcard_manager = cls._dp_wildcard_manager
        parser_config = cls._get_dynamicprompts_parser_config()

        if generator is not None:
            try:
                generated = generator.generate(text, cls.DYNAMIC_RESOLVE_MAX_CANDIDATES) or []
                for value in generated:
                    value = str(value).strip()
                    if value:
                        candidates.append(value)
            except Exception:
                pass

        if wildcard_manager is not None and dp_parse is not None:
            try:
                parsed = dp_parse(text, parser_config=parser_config) if parser_config is not None else dp_parse(text)
                wildcard_names: set[str] = set()
                cls._collect_wildcard_names_from_command(parsed, wildcard_names)
                wildcard_budget = cls.DYNAMIC_RESOLVE_MAX_WILDCARD_VALUES
                for wildcard_name in wildcard_names:
                    if wildcard_budget <= 0:
                        break
                    values = wildcard_manager.get_all_values(wildcard_name)
                    for value in values:
                        value = str(value).strip()
                        if not value:
                            continue
                        candidates.append(value)
                        wildcard_budget -= 1
                        if wildcard_budget <= 0:
                            break
            except Exception:
                pass

        deduped = []
        seen = set()
        for candidate in candidates:
            key = cls._normalize_for_match(candidate)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
            if len(deduped) >= cls.DYNAMIC_RESOLVE_MAX_WILDCARD_VALUES:
                break

        if len(cls._dp_resolve_cache) >= cls.DYNAMIC_RESOLVE_CACHE_MAX_ENTRIES:
            cls._dp_resolve_cache.clear()
        cls._dp_resolve_cache[text] = deduped
        return list(deduped)

    @classmethod
    def _resolve_variant_blocks(cls, text: str, prompt: str):
        def _replace(match: re.Match):
            token = match.group(0)
            options = cls._extract_variant_options(token)
            if not options:
                return token
            best = cls._best_prompt_match(prompt, options)
            return best if best is not None else options[0]

        prev = None
        resolved = text
        loops = 0
        while resolved != prev and loops < 16:
            prev = resolved
            resolved = cls._variant_block_regex.sub(_replace, resolved)
            loops += 1
        return resolved

    @classmethod
    def _resolve_wildcard_tokens(cls, text: str, prompt: str):
        def _replace(match: re.Match):
            wildcard_name = match.group(1)
            candidates = cls._load_wildcard_values(wildcard_name)
            if not candidates:
                return match.group(0)
            best = cls._best_prompt_match(prompt, candidates)
            return best if best is not None else match.group(0)

        return cls._wildcard_token_regex.sub(_replace, text)

    @classmethod
    def _resolve_attention_and_candidates(cls, raw_attention: str, prompt: str):
        attention = (raw_attention or "").strip()
        if not attention:
            return "", []
        if not prompt:
            simple = cls._break_regex.sub(" ", attention).strip()
            return simple, [simple] if simple else []

        resolved = cls._resolve_variant_blocks(attention, prompt)
        resolved = cls._resolve_wildcard_tokens(resolved, prompt)
        resolved = cls._break_regex.sub(" ", resolved)
        resolved = re.sub(r"\s+", " ", resolved).strip(" ,")
        resolved = resolved if resolved else attention

        candidates = cls._attention_candidates(raw_attention, resolved)
        best_dynamic = cls._best_prompt_match(prompt, cls._dynamic_prompt_candidates(attention))
        if best_dynamic:
            resolved = best_dynamic
            candidates = cls._attention_candidates(raw_attention, resolved)
        else:
            best = cls._best_prompt_match(prompt, candidates)
            if best:
                resolved = best
                candidates = cls._attention_candidates(raw_attention, resolved)

        return resolved, candidates

    @classmethod
    def _attention_candidates(cls, raw_attention: str, resolved_attention: str):
        candidates = []
        for term in (resolved_attention, raw_attention):
            term = (term or "").strip()
            if not term:
                continue
            candidates.append(cls._break_regex.sub(" ", term).strip(" ,"))
            for piece in re.split(r",", cls._break_regex.sub(",", term)):
                piece = piece.strip()
                if piece:
                    candidates.append(piece)
        for dynamic_candidate in cls._dynamic_prompt_candidates(raw_attention):
            candidates.append(dynamic_candidate)
            for piece in re.split(r",", cls._break_regex.sub(",", dynamic_candidate)):
                piece = piece.strip()
                if piece:
                    candidates.append(piece)
        deduped = []
        seen = set()
        for candidate in candidates:
            key = cls._normalize_for_match(candidate)
            if candidate and key not in seen:
                seen.add(key)
                deduped.append(candidate)
        return deduped

    @classmethod
    def _sanitize_filename_fragment(cls, text: str):
        text = cls._invalid_filename_chars.sub("_", (text or "").strip())
        text = re.sub(r"\s+", " ", text).strip().strip(".")
        return text or "attention"

    @classmethod
    def _canonicalize_prompt_for_daam(cls, prompt: str):
        """
        Normalize extra-network tag placement for token alignment.
        Forge can expose slightly different chunking when tags (for example LoRA)
        are placed at the start of the prompt, which can destabilize DAAM mapping.
        """
        if prompt is None:
            return ""
        if not isinstance(prompt, str):
            return ""
        prompt = (prompt or "").strip()
        if not prompt:
            return prompt

        tags = cls._extra_network_tag_regex.findall(prompt)
        text_wo_tags = cls._extra_network_tag_regex.sub(" ", prompt)
        text_wo_tags = re.sub(r"\s*,\s*", ", ", text_wo_tags)
        text_wo_tags = re.sub(r",\s*,+", ", ", text_wo_tags)
        text_wo_tags = re.sub(r"\s+", " ", text_wo_tags).strip(" ,")

        if not tags:
            return text_wo_tags

        tag_tail = ", ".join(tag.strip() for tag in tags if tag.strip())
        if not text_wo_tags:
            return tag_tail
        return f"{text_wo_tags}, {tag_tail}"
