import asyncio
import time
from typing import Dict, List


class BenchmarkRunner:
    def __init__(self, agent, retrieval_evaluator, judge, ragas_evaluator=None):
        self.agent = agent
        self.retrieval_evaluator = retrieval_evaluator
        self.judge = judge
        self.ragas_evaluator = ragas_evaluator

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        # 1. Gọi Agent (truyền lịch sử hội thoại nếu là case multi-turn).
        response = await self.agent.query(
            test_case["question"], conversation=test_case.get("conversation")
        )
        latency = time.perf_counter() - start_time

        # 2. Đánh giá Retrieval (chỉ với case có Ground Truth IDs).
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        retrieval = (
            self.retrieval_evaluator.score_case(expected_ids, retrieved_ids)
            if expected_ids else None
        )

        # 3. Chạy RAGAS metrics (faithfulness/relevancy/context precision-recall).
        ragas = None
        if self.ragas_evaluator is not None:
            ragas = await self.ragas_evaluator.score_case(
                test_case["question"], response["answer"],
                response.get("contexts", []), expected_ids, retrieved_ids,
            )

        # 4. Chạy Multi-Judge cho chất lượng câu trả lời.
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"],
        )

        return {
            "ragas": ragas,
            "ragas_cost": ragas["_cost"] if ragas else 0.0,
            "id": test_case.get("id"),
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "tokens_used": response.get("metadata", {}).get("tokens_used", 0),
            "retrieved_ids": retrieved_ids,
            "retrieval": retrieval,
            "judge": judge_result,
            "judge_cost": judge_result.get("judge_cost", 0.0),
            "judge_tokens_in": judge_result.get("judge_tokens_in", 0),
            "judge_tokens_out": judge_result.get("judge_tokens_out", 0),
            "metadata": test_case.get("metadata", {}),
            "status": "fail" if judge_result["final_score"] < 3 else "pass",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Chạy song song bằng asyncio.gather, giới hạn batch_size để tránh Rate Limit."""
        results = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            tasks = [self.run_single_test(case) for case in batch]
            results.extend(await asyncio.gather(*tasks))
        return results
