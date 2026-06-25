# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Văn Sáng — AI20K
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 850, rounded up)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: $0.10 (~17 phút @ $0.35/hr)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|------------------|------------|-----------|-----------|------------|
| 8    | ~6,553,600       | 5.2 min    | 8.1 GB    | 1.35      | 3.85       |
| 16   | ~13,107,200      | 5.4 min    | 8.2 GB    | 1.31      | 3.70       |
| 64   | ~52,428,800      | 6.1 min    | 8.8 GB    | 1.28      | 3.59       |
| Base | -                | -          | -         | 1.55      | 4.71       |

## 3. Loss Curve Analysis
[Đính kèm loss_curve.png]
- **Quan sát**: Không có hiện tượng overfitting rõ rệt. Train loss giảm ổn định và tiệm cận dần, eval loss song song giảm nhẹ và ổn định quanh mức 1.30. Điều này chứng tỏ mô hình học được format mà không bị học vẹt.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là... (hơi cứng nhắc, thỉnh thoảng lẫn từ tiếng Anh)
**Fine-tuned (r=16)**: Học máy (Machine learning) là một nhánh của trí tuệ nhân tạo, cho phép máy tính tự học từ dữ liệu thay vì được lập trình cụ thể... (giọng văn tự nhiên, mượt mà hơn).
**Nhận xét**: Improved, mô hình trả lời giống một trợ lý người Việt hơn.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Đây là mã Python... (code đúng, nhưng thiếu comments tiếng Việt)
**Fine-tuned (r=16)**: Dưới đây là hàm Python để tính số Fibonacci thứ n sử dụng đệ quy có nhớ (memoization)... (giải thích rõ ràng, code chuẩn, biến dễ hiểu)
**Nhận xét**: Improved, cấu trúc câu trả lời tốt hơn.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Đơn giản 2. Dễ sử dụng... (Liệt kê ngắn gọn)
**Fine-tuned (r=16)**: 5 nguyên tắc thiết kế UI/UX cơ bản: 1. Đơn giản hoá giao diện, 2. Tính nhất quán, 3. Phản hồi cho người dùng, 4. Dễ điều hướng, 5. Phù hợp với đối tượng mục tiêu. Mỗi nguyên tắc đều có giải thích chi tiết.
**Nhận xét**: Improved, format list có đầu đuôi và giải thích.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: Trả lời chung chung, có lúc nhầm sang kiến trúc model.
**Fine-tuned (r=16)**: LoRA thêm các low-rank adapters vào model gốc, trong khi QLoRA kết hợp LoRA với quantization 4-bit (NF4) cho base model để tiết kiệm VRAM hơn nữa.
**Nhận xét**: Improved, đáp án chính xác và focus đúng vào instruction.

### Example 5
**Prompt**: List 3 câu hỏi phỏng vấn cho ML Engineer junior.
**Base**: Trả lời bằng tiếng Anh.
**Fine-tuned (r=16)**: Trả lời bằng tiếng Việt chuyên ngành: 1. Overfitting là gì và cách phòng tránh? 2. Sự khác biệt giữa random forest và gradient boosting? 3. Cách bạn xử lý dữ liệu bị mất (missing values)?
**Nhận xét**: Improved, align đúng ngôn ngữ yêu cầu (tiếng Việt).

## 5. Conclusion về Rank Trade-off

Dựa trên thực nghiệm, rank `r=16` mang lại ROI tốt nhất. Nó cân bằng giữa thời gian huấn luyện (~5.4 phút), VRAM tiêu thụ (~8.2 GB) và chất lượng (Perplexity giảm từ 4.71 xuống 3.70). 
Khi tăng rank lên `r=64`, ta bắt đầu thấy hiện tượng diminishing returns: số lượng tham số tăng gấp 4 lần, nhưng Perplexity chỉ giảm rất nhẹ (từ 3.70 xuống 3.59), trong khi tốn thêm bộ nhớ và thời gian tính toán. 
Recommendation: Nếu deploy lên production cho task general tiếng Việt này, tôi sẽ chọn `r=16` vì nó tối ưu chi phí phục vụ mà vẫn đảm bảo được chất lượng generation mượt mà.

## 6. What I Learned
- **Bản chất của QLoRA**: Hiểu rõ sự kết hợp của 4-bit quantization và LoRA adapters, giúp giải quyết bài toán VRAM khi fine-tune model >3 tỷ tham số trên Colab T4.
- **Tầm quan trọng của dữ liệu**: Mặc dù dataset nhỏ (~200 samples) nhưng chất lượng instruction (định dạng Alpaca) rất quan trọng để LLM bắt chước format.
- **Rank selection**: Không phải rank càng cao càng tốt, việc đo lường ROI giữa memory/time và perplexity là kỹ năng quan trọng trong MLOps.
