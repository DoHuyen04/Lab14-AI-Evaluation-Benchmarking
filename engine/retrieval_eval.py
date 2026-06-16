import json
import os
import re
from typing import Dict, List, Optional


def _tokens(text: str) -> set:
    return {t for t in re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE) if len(t) > 1}


class RetrievalEvaluator:
    """Đánh giá chất lượng tầng Retrieval bằng Hit Rate và MRR.

    Dùng Ground Truth `expected_retrieval_ids` (từ Golden Set) so với
    `retrieved_ids` mà Agent trả về.
    """

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str],
                           top_k: Optional[int] = None) -> float:
        """1.0 nếu có ít nhất 1 expected_id nằm trong top_k retrieved_ids."""
        k = top_k or self.top_k
        top_retrieved = retrieved_ids[:k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """Mean Reciprocal Rank: 1 / vị trí (1-indexed) của expected_id đầu tiên."""
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    def score_case(self, expected_ids: List[str], retrieved_ids: List[str]) -> Dict:
        """Tính metrics cho 1 test case."""
        return {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids[:self.top_k],
        }

    def aggregate(self, per_case_scores: List[Dict]) -> Dict:
        """Tổng hợp Hit Rate / MRR trung bình trên các case ĐÃ có Ground Truth.

        Các case out-of-context / red-team (expected rỗng) được loại trừ vì
        không phải bài toán retrieval.
        """
        scores = [s for s in per_case_scores if s is not None]
        n = len(scores)
        if n == 0:
            return {"hit_rate": 0.0, "mrr": 0.0, "evaluated_cases": 0}
        return {
            "hit_rate": sum(s["hit_rate"] for s in scores) / n,
            "mrr": sum(s["mrr"] for s in scores) / n,
            "evaluated_cases": n,
        }


# Bảng giá OpenAI (USD / 1K token) cho phần RAGAS.
_RAGAS_PRICING = {"gpt-4o-mini": (0.00015, 0.00060), "gpt-4o": (0.00250, 0.01000)}

_RAGAS_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number"},
        "answer_relevancy": {"type": "number"},
    },
    "required": ["faithfulness", "answer_relevancy"],
    "additionalProperties": False,
}


class RagasEvaluator:
    """RAGAS-style metrics (custom, hướng B) cho tầng Generation:

      - faithfulness: tỉ lệ thông tin trong câu trả lời được context hỗ trợ (chống bịa).
      - answer_relevancy: mức câu trả lời bám đúng trọng tâm câu hỏi.
      - context_precision / context_recall: dựa trên Ground Truth retrieval IDs.

    faithfulness & answer_relevancy chấm bằng LLM (1 lời gọi/case), có fallback
    offline (độ trùng từ) để chạy được khi thiếu OPENAI_API_KEY.
    """

    def __init__(self, model: str = None):
        self.model = model or os.environ.get("RAGAS_MODEL", "gpt-4o-mini")
        self._online = bool(os.environ.get("OPENAI_API_KEY"))
        self._client = None
        if self._online:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI()
            except ImportError:
                self._online = False

    def _context_metrics(self, expected_ids: List[str], retrieved_ids: List[str]):
        """Context Precision/Recall — chỉ tính khi có Ground Truth."""
        if not expected_ids:
            return None, None
        inter = set(retrieved_ids) & set(expected_ids)
        precision = len(inter) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(inter) / len(expected_ids)
        return precision, recall

    def _heuristic_gen(self, question: str, answer: str, contexts: List[str]):
        ans = _tokens(answer)
        ctx = set().union(*[_tokens(c) for c in contexts]) if contexts else set()
        q = _tokens(question)
        faith = (len(ans & ctx) / len(ans)) if (ans and ctx) else None
        relev = (len(ans & q) / len(q)) if q else 0.0
        return faith, min(1.0, relev), {"in": 0, "out": 0, "cost": 0.0}

    async def _llm_gen(self, question: str, answer: str, contexts: List[str]):
        context_text = "\n".join(contexts) if contexts else "(không có context)"
        system = (
            "Bạn là công cụ đánh giá RAGAS. Trả về 2 số trong [0,1]:\n"
            "- faithfulness: tỉ lệ nội dung câu trả lời được CONTEXT hỗ trợ (1.0 = hoàn toàn "
            "dựa trên context, không bịa). Nếu không có context và câu trả lời là từ chối/"
            "nói không biết thì cho 1.0.\n"
            "- answer_relevancy: mức câu trả lời bám đúng trọng tâm CÂU HỎI."
        )
        user = f"CÂU HỎI:\n{question}\n\nCONTEXT:\n{context_text}\n\nCÂU TRẢ LỜI:\n{answer}"
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "ragas", "strict": True,
                                                 "schema": _RAGAS_SCHEMA}},
            )
            data = json.loads(resp.choices[0].message.content)
            t_in, t_out = resp.usage.prompt_tokens, resp.usage.completion_tokens
            p_in, p_out = _RAGAS_PRICING.get(self.model, (0.0, 0.0))
            cost = t_in / 1000 * p_in + t_out / 1000 * p_out
            faith = max(0.0, min(1.0, float(data["faithfulness"])))
            relev = max(0.0, min(1.0, float(data["answer_relevancy"])))
            return faith, relev, {"in": t_in, "out": t_out, "cost": cost}
        except Exception:  # noqa: BLE001 - rơi về heuristic nếu lỗi API
            return self._heuristic_gen(question, answer, contexts)

    async def score_case(self, question: str, answer: str, contexts: List[str],
                         expected_ids: List[str], retrieved_ids: List[str]) -> Dict:
        if self._online:
            faith, relev, usage = await self._llm_gen(question, answer, contexts)
        else:
            faith, relev, usage = self._heuristic_gen(question, answer, contexts)
        precision, recall = self._context_metrics(expected_ids, retrieved_ids)
        return {
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": precision,
            "context_recall": recall,
            "_cost": usage["cost"],
        }

    def aggregate(self, per_case: List[Dict]) -> Dict:
        def mean(key):
            vals = [c[key] for c in per_case if c.get(key) is not None]
            return sum(vals) / len(vals) if vals else 0.0
        return {
            "faithfulness": mean("faithfulness"),
            "answer_relevancy": mean("answer_relevancy"),
            "context_precision": mean("context_precision"),
            "context_recall": mean("context_recall"),
        }
