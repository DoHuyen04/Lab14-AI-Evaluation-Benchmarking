# Reflection cá nhân — Đỗ Huyền (MSSV: 2A202600880)

**Vai trò trong nhóm:** Thực hiện toàn bộ dự án (làm cá nhân) — Data/SDG, Eval Engine, Multi-Judge, Regression & Báo cáo.
**Các module phụ trách chính:** `data/synthetic_gen.py`, `data/knowledge_base.json`, `agent/main_agent.py`, `engine/retrieval_eval.py`, `engine/llm_judge.py`, `engine/runner.py`, `main.py`, `analysis/failure_analysis.md`.

---

## 1. Engineering Contribution (15đ)

- **Golden Dataset & SDG** (`data/synthetic_gen.py` + `data/knowledge_base.json`): xây knowledge base 16 tài liệu có ID chunk làm Ground Truth, và sinh **54 test case** phủ đủ các nhóm khó (factual, multi-hop, out-of-context, ambiguous, conflicting, prompt injection, goal hijacking, multi-turn, latency, cost). Có hàm `validate_cases` chặn trùng/sai ID và bắt buộc ≥50 case; có chế độ `--augment` sinh thêm bằng OpenAI.
- **Agent RAG thật** (`agent/main_agent.py`): viết `TfidfRetriever` (TF-IDF + cosine, không cần thư viện ngoài) trả `retrieved_ids` thật; tầng Generation gọi LLM với **System Prompt + guardrail** (chỉ trả lời theo context, từ chối injection, biết nói "không biết"). Có fallback offline.
- **Eval Engine**:
  - `engine/retrieval_eval.py`: `RetrievalEvaluator` (Hit Rate, MRR) + `RagasEvaluator` (Faithfulness, Answer Relevancy, Context Precision/Recall).
  - `engine/llm_judge.py`: **Multi-Judge** gọi 2 model thật, `cohens_kappa`, xử lý xung đột bằng arbiter, `check_position_bias`, theo dõi token & chi phí.
  - `engine/runner.py`: Async Runner chạy song song theo batch (`asyncio.gather`).
- **Regression & Cost** (`main.py`): so sánh V1 (top_k=1) vs V2 (top_k=3) và **Release Gate** 4 tiêu chí (Quality/Retrieval/Cost/Latency).

**Kết quả đo được (reports/summary.json):** Avg Judge Score **4.60/5**, Hit Rate **92.3%**, MRR **0.803**, Faithfulness **0.99**, Agreement **96.3%**, Cohen's Kappa **0.54**; Release Gate → **APPROVE**.

## 2. Technical Depth (15đ)

- **MRR (Mean Reciprocal Rank):** trung bình của `1/vị trí` chunk đúng đầu tiên trong danh sách trả về. Em dùng kèm Hit Rate vì Hit Rate chỉ trả lời "có/không nằm trong top-k", còn MRR **thưởng cho việc xếp chunk đúng lên cao hơn**. Trong dự án Hit Rate 0.92 nhưng MRR 0.80 → đa số tìm đúng nhưng chưa luôn ở vị trí 1.
- **Cohen's Kappa:** đo độ đồng thuận giữa 2 judge **sau khi loại trừ phần trùng do may rủi** (`κ = (po − pe)/(1 − pe)`). Cần nó ngoài Agreement Rate thô vì khi điểm dồn về vài giá trị, hai judge có thể "trùng nhau ngẫu nhiên" làm agreement bị thổi phồng. Dự án đạt κ ≈ 0.54 (đồng thuận mức trung bình) dù agreement thô tới 96%.
- **Position Bias:** thiên hướng judge ưu ái câu trả lời theo **vị trí** (A trước/B trước) thay vì theo chất lượng. Em kiểm tra bằng `LLMJudge.check_position_bias`: chấm cặp A/B theo cả 2 thứ tự, nếu "người thắng" đổi khi đảo vị trí thì kết luận có thiên vị.
- **Trade-off Chi phí ↔ Chất lượng:** tăng top_k 1→3 (V1→V2) làm **Recall tăng 0.63→0.89 nhưng Precision giảm 0.72→0.34**, và context nhiễu hơn nên chi phí judge cao hơn. Em đề xuất **giảm ~30% chi phí eval** bằng cascade: chạy 1 judge rẻ (`gpt-4o-mini`) mặc định, chỉ escalate `gpt-4o` khi điểm gần ngưỡng pass/fail hoặc 2 judge bất đồng.

## 3. Problem Solving (10đ)

- **Vấn đề 1 — Judge bất đồng >1 điểm:** ban đầu lấy trung bình che mất xung đột. Em thêm **arbiter**: khi lệch quá ngưỡng thì gọi judge thứ 3 và lấy **trung vị**; offline thì lấy điểm bảo thủ (thấp nhất). Theo dõi `judge_conflicts` để giải trình.
- **Vấn đề 2 — Agent fail toàn bộ red_team/edge:** agent ban đầu chỉ ghép context (không LLM) nên bị prompt injection và không biết từ chối. Em nâng cấp Generation bằng LLM + System Prompt guardrail → nhóm `red_team` từ **2.43 (yếu nhất) lên 4.93 (tốt nhất)**, số fail giảm 13→4.
- **Vấn đề 3 — Chấm bài không có API key:** mọi cấu phần gọi LLM (SDG, judge, ragas, agent) đều có **fallback offline** (heuristic theo độ trùng từ) để pipeline luôn chạy được khi chấm.
- **Bài học:** (1) chất lượng generation **bị chặn trên bởi retrieval** — 3/4 case fail còn lại đều do retrieval sai; (2) một Release Gate tốt phải đa tiêu chí, không duyệt mù chỉ vì điểm chất lượng tăng; (3) judge LLM ngẫu nhiên nên cần báo cáo khoảng dao động thay vì một con số tuyệt đối.

---

## (Tự đánh giá) Bảng đối chiếu đóng góp

| Hạng mục                      | Tự chấm | Bằng chứng                                                             |
| ----------------------------- | :-----: | ---------------------------------------------------------------------- |
| Engineering Contribution (15) |  14/15  | Toàn bộ module trong `engine/`, `agent/`, `data/`, `main.py`           |
| Technical Depth (15)          |  14/15  | Giải thích MRR/Kappa/Position Bias/trade-off ở mục 2, gắn số liệu thật |
| Problem Solving (10)          |  9/10   | Arbiter xung đột, nâng cấp guardrail, fallback offline (mục 3)         |
