# Báo cáo Phân tích Thất bại (Failure Analysis Report)

> Số liệu trích từ `reports/benchmark_results.json` và `reports/summary.json`
> (phiên bản **Agent_V2_Optimized** — RAG-LLM thật: TF-IDF retrieval + Generation
> bằng LLM có guardrail, top_k=3, 54 test cases). Judge LLM có yếu tố ngẫu nhiên
> nên con số tuyệt đối dao động nhẹ giữa các lần chạy; xu hướng và cụm lỗi ổn định.

## 1. Tổng quan Benchmark
- **Tổng số cases:** 54
- **Tỉ lệ Pass/Fail:** 50 / 4 (Pass 92.6%) — ngưỡng fail: điểm Judge < 3.0
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.99
    - Answer Relevancy: 0.84
    - Context Precision: 0.34
    - Context Recall: 0.89
- **Retrieval:** Hit Rate 92.3% | MRR 0.803 (trên 39 case có Ground Truth)
- **Điểm LLM-Judge trung bình:** 4.60 / 5.0
- **Độ tin cậy Judge:** Agreement 96.3% | Cohen's Kappa 0.54 | 2 case xung đột (auto-resolve bằng arbiter)

### Điểm Judge trung bình theo nhóm
| Nhóm (category) | Số case | Điểm TB | Nhận xét |
|---|:---:|:---:|---|
| edge | 11 | 4.23 | Đã cải thiện mạnh — biết nói "không biết" / từ chối; ambiguous còn yếu (Case #3) |
| technical | 4 | 4.25 | Tốt; latency_stress kéo xuống (xem Case #2) |
| multi_turn | 5 | 4.30 | Tốt; lỗi correction kéo xuống (Case #1) |
| factual | 27 | 4.78 | Rất tốt |
| red_team | 7 | **4.93** | **Tốt nhất** — guardrail chặn injection/hijacking hiệu quả |

> **So với phiên bản agent cũ (chỉ ghép context, không LLM):** điểm trung bình tăng
> từ 3.74 → 4.60, số fail giảm từ 13 → 4, và nhóm `red_team` từ yếu nhất (2.43)
> trở thành mạnh nhất (4.93). Việc thêm **System Prompt + guardrail** ở tầng
> Generation đã xử lý gần hết các lỗi an toàn/biên.

## 2. Phân nhóm lỗi (Failure Clustering)
Chỉ còn **4 case fail**, và đều quy về **chất lượng Retrieval / xử lý đầu vào / clarify**
— không còn lỗi an toàn:

| Nhóm lỗi | Số lượng | Case | Nguyên nhân gốc |
|---|:---:|---|---|
| **Retrieval sai do Chunking** | 2 | c049, c006 | Chunk bảng giá gộp nhiều gói → loãng từ khóa; `doc_promo` gây nhiễu |
| **Loãng truy vấn do input dài** | 1 | c051 | Đoạn văn bản dài lấn át câu hỏi thật khi tính TF-IDF |
| **Không hỏi lại khi mơ hồ** | 1 | c035 | Agent đoán một nghĩa thay vì hỏi làm rõ |

*(3 trong 4 case fail có Hit Rate = 0 → lỗi gốc nằm ở tầng Retrieval.)*

> **Quan sát quan trọng — Retrieval ↔ Answer Quality:** cả 3 case fail đều là case mà
> **retrieval trả về sai chunk** (Hit Rate = 0) hoặc đầu vào gây nhiễu retrieval.
> Khi tầng sinh đã tốt (Faithfulness 0.98), lỗi còn lại **dịch chuyển hoàn toàn về
> tầng Retrieval** — đúng nguyên lý "chất lượng generation bị chặn trên bởi chất
> lượng retrieval". Đồng thời khi tăng top_k 1→3 (V1→V2): Recall tăng 0.63→0.89
> nhưng Precision giảm 0.72→0.34 (đánh đổi precision/recall điển hình).

## 3. Phân tích 5 Whys (3 case fail, mỗi case một tầng lỗi khác nhau)

### Case #1 — c049 "À nhầm, tôi dùng gói Free. Vậy dung lượng của tôi là bao nhiêu?" (Chunking/Retrieval)
- **Symptom:** Điểm 2.0, Hit Rate = 0. Expected `doc_billing#0`, nhưng retriever trả về `doc_trial#0`, `doc_storage#0`, `doc_promo#0`.
1. **Why 1:** Vì chunk chứa "Free = 5GB" (`doc_billing#0`) không lọt top-3.
2. **Why 2:** Vì các từ trong hội thoại ("dùng thử", "dung lượng") khớp mạnh hơn với `doc_trial`/`doc_storage`/`doc_promo`.
3. **Why 3:** Vì `doc_billing#0` gộp cả 3 gói trong một chunk → tín hiệu cho riêng "Free/5GB" bị pha loãng.
4. **Why 4:** Vì chiến lược chunking "mỗi mục = 1 chunk" tạo chunk đa-chủ-đề, rộng.
5. **Why 5:** Vì ingestion chưa tách bảng giá thành đơn vị truy xuất theo từng gói.
- **Root Cause:** **Chiến lược Chunking không phù hợp dữ liệu bảng biểu** — chunk đa-chủ-đề làm loãng tín hiệu truy xuất.

### Case #2 — c051 "(đoạn log rất dài)... Bộ phận hỗ trợ làm việc mấy giờ?" (Ingestion/Xử lý đầu vào)
- **Symptom:** Điểm 2.0, Hit Rate = 0. Expected `doc_support#0`, trả về `doc_export#0`, `doc_2fa#0`, `doc_password#0`.
1. **Why 1:** Vì câu hỏi thật ("giờ hỗ trợ") bị chìm trong khối văn bản chèn dài.
2. **Why 2:** Vì retriever dùng toàn bộ input làm truy vấn, từ nhiễu chiếm trọng số.
3. **Why 3:** Vì không có bước tách/đề cao "câu hỏi thực" khỏi ngữ cảnh dán kèm.
4. **Why 4:** Vì agent coi mọi input là truy vấn nguyên khối, không tiền xử lý độ dài.
5. **Why 5:** Vì pipeline thiếu bước chuẩn hóa input (cắt bớt/tách câu hỏi/tóm tắt trước khi retrieve).
- **Root Cause:** **Thiếu tiền xử lý truy vấn** — input dài làm loãng tín hiệu; cũng là rủi ro latency/cost.

### Case #3 — c035 "Giới hạn là bao nhiêu vậy?" (Prompting — chưa hỏi lại)
- **Symptom:** Điểm 2.5. Câu hỏi mơ hồ (dung lượng? API? chia sẻ?) nhưng agent đoán một nghĩa (API) thay vì hỏi lại.
1. **Why 1:** Vì agent chọn ngay một cách hiểu thay vì làm rõ ý định.
2. **Why 2:** Vì retriever đã trả về context API → agent "thấy có dữ liệu" nên trả lời luôn.
3. **Why 3:** Vì System Prompt yêu cầu "hỏi lại khi mơ hồ" nhưng không định nghĩa rõ *khi nào* coi là mơ hồ.
4. **Why 4:** Vì không có bước phát hiện nhập nhằng (vd nhiều chủ đề cùng khớp) trước khi sinh.
5. **Why 5:** Vì thiết kế chưa tách riêng bước "phân loại ý định / kiểm tra đủ thông tin".
- **Root Cause:** **Thiếu cơ chế phát hiện & xử lý câu hỏi mơ hồ** ở tầng Prompting (clarify chưa đủ chặt).

## 4. Kế hoạch cải tiến (Action Plan) — gắn với Root Cause
- [ ] **(Chunking — Case #1)** Tách `doc_billing` theo từng gói cước; thử Semantic Chunking thay "mỗi mục = 1 chunk".
- [ ] **(Retrieval — Case #1)** Thêm bước **Reranking** sau TF-IDF (hoặc dùng embedding) để đẩy chunk đúng lên top-k; giảm nhiễu từ `doc_promo`.
- [ ] **(Ingestion — Case #2)** Tiền xử lý input: tách câu hỏi khỏi ngữ cảnh dán kèm hoặc giới hạn độ dài truy vấn trước khi retrieve.
- [ ] **(Prompting — Case #3)** Bổ sung quy tắc clarify rõ ràng hơn (nêu vài chủ đề cùng khớp → hỏi lại); cân nhắc bước phát hiện nhập nhằng.
- [ ] **(Đo lại)** Sau cải tiến, chạy lại benchmark; mục tiêu: 4 case fail còn lại pass mà vẫn giữ Faithfulness ≥ 0.98 và Hit Rate ≥ 0.92.
