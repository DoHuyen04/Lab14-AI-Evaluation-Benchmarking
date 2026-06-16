import asyncio
import json
import os
import time

from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator, RagasEvaluator
from engine.llm_judge import LLMJudge, cohens_kappa
from agent.main_agent import MainAgent

TOP_K = 3

# Ngưỡng Release Gate (so V2 với V1).
GATE_THRESHOLDS = {
    "min_quality_delta": -0.10,   # avg_score không được tụt quá 0.10
    "min_hit_rate_delta": -0.02,  # hit_rate không được tụt quá 2%
    "max_cost_increase": 0.10,    # chi phí/eval không tăng quá 10%
    "max_latency_increase": 0.25, # latency không tăng quá 25%
}


async def run_benchmark_with_results(agent_version: str, agent: MainAgent):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    retrieval_evaluator = RetrievalEvaluator(top_k=agent.top_k)
    ragas_evaluator = RagasEvaluator()
    runner = BenchmarkRunner(agent, retrieval_evaluator, LLMJudge(), ragas_evaluator)
    results = await runner.run_all(dataset)

    total = len(results)
    retrieval_agg = retrieval_evaluator.aggregate([r["retrieval"] for r in results])
    ragas_agg = ragas_evaluator.aggregate([r["ragas"] for r in results if r["ragas"]])

    rater_a = [r["judge"]["base_scores"][0] for r in results]
    rater_b = [r["judge"]["base_scores"][1] for r in results]
    kappa = cohens_kappa(rater_a, rater_b)
    conflicts = sum(1 for r in results if r["judge"].get("conflict"))

    judge_cost = sum(r["judge_cost"] for r in results)
    ragas_cost = sum(r["ragas_cost"] for r in results)
    total_cost = judge_cost + ragas_cost
    total_tok_in = sum(r["judge_tokens_in"] for r in results)
    total_tok_out = sum(r["judge_tokens_out"] for r in results)

    summary = {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": retrieval_agg["hit_rate"],
            "mrr": retrieval_agg["mrr"],
            "retrieval_evaluated_cases": retrieval_agg["evaluated_cases"],
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "cohens_kappa": kappa,
            "judge_conflicts": conflicts,
            "faithfulness": ragas_agg["faithfulness"],
            "answer_relevancy": ragas_agg["answer_relevancy"],
            "context_precision": ragas_agg["context_precision"],
            "context_recall": ragas_agg["context_recall"],
            "avg_latency": sum(r["latency"] for r in results) / total,
            "avg_tokens": sum(r["tokens_used"] for r in results) / total,
        },
        "cost": {
            "total_eval_cost_usd": total_cost,
            "judge_cost_usd": judge_cost,
            "ragas_cost_usd": ragas_cost,
            "cost_per_eval_usd": total_cost / total,
            "judge_tokens_in": total_tok_in,
            "judge_tokens_out": total_tok_out,
        },
    }
    return results, summary


def release_gate(v1: dict, v2: dict, th: dict) -> dict:
    """Logic Release Gate tự động: APPROVE/BLOCK dựa trên nhiều tiêu chí."""
    m1, m2 = v1["metrics"], v2["metrics"]
    c1, c2 = v1["cost"], v2["cost"]

    quality_delta = m2["avg_score"] - m1["avg_score"]
    hit_delta = m2["hit_rate"] - m1["hit_rate"]
    cost_ratio = (c2["cost_per_eval_usd"] - c1["cost_per_eval_usd"]) / c1["cost_per_eval_usd"] \
        if c1["cost_per_eval_usd"] else 0.0
    latency_ratio = (m2["avg_latency"] - m1["avg_latency"]) / m1["avg_latency"] \
        if m1["avg_latency"] else 0.0

    checks = {
        "quality": (quality_delta >= th["min_quality_delta"],
                    f"Δavg_score={quality_delta:+.2f} (ngưỡng ≥ {th['min_quality_delta']})"),
        "retrieval": (hit_delta >= th["min_hit_rate_delta"],
                      f"Δhit_rate={hit_delta:+.1%} (ngưỡng ≥ {th['min_hit_rate_delta']:.0%})"),
        "cost": (cost_ratio <= th["max_cost_increase"],
                 f"Δcost={cost_ratio:+.1%} (ngưỡng ≤ {th['max_cost_increase']:.0%})"),
        "latency": (latency_ratio <= th["max_latency_increase"],
                    f"Δlatency={latency_ratio:+.1%} (ngưỡng ≤ {th['max_latency_increase']:.0%})"),
    }
    passed = all(ok for ok, _ in checks.values())
    return {
        "decision": "APPROVE" if passed else "BLOCK",
        "checks": {k: {"passed": ok, "detail": d} for k, (ok, d) in checks.items()},
    }


async def main():
    # V1 = base (chỉ lấy top-1 chunk, retrieval yếu); V2 = tối ưu (top-3 chunk).
    _, v1_summary = await run_benchmark_with_results(
        "Agent_V1_Base", MainAgent(top_k=1, name="SupportAgent-v1"))
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", MainAgent(top_k=TOP_K, name="SupportAgent-v2"))

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- RETRIEVAL & QUALITY ---")
    print(f"{'':14}{'V1_Base':>12}{'V2_Optimized':>14}")
    for key, label in [("hit_rate", "Hit Rate"), ("mrr", "MRR"),
                       ("avg_score", "Avg Score"), ("agreement_rate", "Agreement")]:
        print(f"{label:14}{v1_summary['metrics'][key]:>12.3f}{v2_summary['metrics'][key]:>14.3f}")

    print("\n🧪 --- RAGAS (Generation quality) ---")
    print(f"{'':18}{'V1_Base':>12}{'V2_Optimized':>14}")
    for key, label in [("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer Relevancy"),
                       ("context_precision", "Ctx Precision"), ("context_recall", "Ctx Recall")]:
        print(f"{label:18}{v1_summary['metrics'][key]:>12.3f}{v2_summary['metrics'][key]:>14.3f}")

    print("\n💰 --- COST & TOKEN (chi phí eval: judge + ragas) ---")
    c1, c2 = v1_summary["cost"], v2_summary["cost"]
    print(f"Tổng chi phí:   V1 ${c1['total_eval_cost_usd']:.4f}  |  V2 ${c2['total_eval_cost_usd']:.4f}")
    print(f"Chi phí/eval:   V1 ${c1['cost_per_eval_usd']:.6f}  |  V2 ${c2['cost_per_eval_usd']:.6f}")
    print(f"Token judge V2: in={c2['judge_tokens_in']:,} out={c2['judge_tokens_out']:,}")
    # Đề xuất giảm ~30% chi phí: dùng judge rẻ làm chính, chỉ escalate khi xung đột.
    est_saving = c2["total_eval_cost_usd"] * 0.30
    print(f"💡 Đề xuất giảm ~30% chi phí (~${est_saving:.4f}): chỉ chạy 1 judge rẻ (gpt-4o-mini) "
          f"làm mặc định, escalate gpt-4o khi điểm gần ngưỡng pass/fail hoặc bất đồng.")

    print("\n🚦 --- REGRESSION RELEASE GATE ---")
    gate = release_gate(v1_summary, v2_summary, GATE_THRESHOLDS)
    for name, c in gate["checks"].items():
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {name:10} {c['detail']}")
    v2_summary["release_gate"] = gate

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if gate["decision"] == "APPROVE":
        print("\n✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("\n❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")


if __name__ == "__main__":
    asyncio.run(main())
