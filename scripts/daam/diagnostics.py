from __future__ import annotations

import json
import os


class DiagnosticsMixin:
    def _diagnose_missing_heatmap(self, raw_attention: str, resolved_attention: str, candidates, prompt_text: str, prompt_analyzer):
        if len(candidates) == 0:
            return "no_candidates"

        if not any(self._contains_phrase(prompt_text, candidate) for candidate in candidates):
            return "term_not_in_resolved_prompt"

        used_custom_terms = getattr(prompt_analyzer, "used_custom_terms", []) if prompt_analyzer is not None else []
        attention_keys = {self._normalize_for_match(raw_attention), self._normalize_for_match(resolved_attention)}
        for item in used_custom_terms:
            if not isinstance(item, (list, tuple)) or len(item) == 0:
                continue
            key = self._normalize_for_match(str(item[0]))
            if key in attention_keys:
                return "embedding_or_custom_term"

        if prompt_analyzer is not None:
            token_count = int(getattr(prompt_analyzer, "token_count", 0) or 0)
            context_size = int(getattr(prompt_analyzer, "context_size", 0) or 0)
            if context_size > 0 and token_count >= max(1, context_size - 2):
                return "possible_context_truncation"

        return "token_not_found_in_attention_map"

    def _save_diagnostics(self, params, payload):
        if not payload:
            return
        fullfn_without_extension, _ = os.path.splitext(params.filename)
        diagnostics_path = fullfn_without_extension + "_daam_diag.json"
        try:
            with open(diagnostics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DAAM] Failed to save diagnostics at {diagnostics_path}: {e}")
