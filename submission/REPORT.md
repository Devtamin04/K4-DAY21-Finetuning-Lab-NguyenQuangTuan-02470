# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Quang Tuấn  **MSSV**: 02470  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 14.6 GB (Colab, kết nối qua VS Code) · fp16 (T4 không có bf16)

> Mọi con số dưới đây lấy từ `results/` của một lần chạy đầy đủ NB1 → NB5
> (`EVAL_LIMIT` không đặt, `EPOCHS=2`, commit `d27c1c0`, tổng 49,1 phút).

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định tier T4) | Model lớn nhất vừa LoRA 16-bit trên T4 free (peak 8,78 GB / 14,6 GB). Giữ mặc định để mask và template đã được lab kiểm chứng, và để số đo so được với `docs/MEASURED-T4-2026-08-20.md`. |
| Dataset | 250 ticket CSKH tiếng Việt → JSON 4 trường (mặc định) | Có thang chấm khách quan (accuracy từng trường, JSON parse) nên không cần LLM judge. Lần chạy đầu dùng corpus gốc để checksum tập eval giữ nguyên, phép so sánh không bị nghi ngờ. |
| Prompt (b) | `OPTIMIZED_PROMPT` gốc, **không sửa** | SHA `719e74d3b6232053`, `make verify` xác nhận không bị đổi. |

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage; eval: 50 target + 15 regression |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 (theo tier) — p95 đo được là **98**, p99 = 100, max = 101; NB1 gợi ý 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / **30** optimizer step (batch 1 × grad-accum 16 = batch hiệu dụng 16) |
| LoRA | all-linear (`text-linear`, 12 module) · r=16 · α=32 · LR 1e-4 · cosine · warmup 3 |

**Về `max_length`:** NB1 cảnh báo p95 gợi ý 256 nhưng tier đặt 1024. Tôi giữ 1024 vì với
dữ liệu này hai giá trị cho **kết quả huấn luyện y hệt nhau**: mẫu dài nhất chỉ 101 token,
nên không mẫu nào bị cắt ở cả 256 lẫn 1024. Với batch 1 và không packing, cũng không có
token padding thừa, nên `max_length` lớn hơn không tốn thêm VRAM hay thời gian. Nếu đổi sang
dataset có câu trả lời dài hơn (ví dụ có reasoning trace), tôi sẽ đặt lại theo p95 đo được.

**Template có giữ khối `<think>` không?** **Có.** `template_check.json`: `open_tag_present: true`,
`body_present: true`, verdict *"reasoning preserved — safe to train on traces"*. Chuỗi render:
`<|im_start|>assistant\n<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>\n\n4<|im_end|>`.
Với corpus này, câu trả lời huấn luyện là JSON trần, nên generation prompt chỉ chứa khối
`<think>` rỗng (xem mask proof bên dưới). Vì vậy không có reasoning trace nào để xử lý, và
`valid_trace_rate = 0.0` ở NB5 là đúng kỳ vọng, không phải lỗi.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39 / 94 token ở mẫu minh hoạ; toàn tập train: 9014 / 20951 = 43,0%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn **được tính loss** (`assistant-only`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị mask** (không tính loss):

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>
```

Đối chứng: cùng mẫu với `MASK_MODE=everything` cho 94/94 token (100%) được tính loss, tức là
model học cả việc viết lại system prompt và câu hỏi. Kết quả 41% cho thấy mask cắt đúng tại
ranh giới assistant. Còn token `</think>` và `<|im_end|>` nằm trong loss, nên model học được
cả cách đóng khối think lẫn cách dừng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3311.4 |
| (b) base + optimized prompt | **0.765** | 0.791 | 1.000 | 1030.6 |
| (c) LoRA fine-tune | **0.970** | **0.567** | 1.000 | 1367.1 |

*(a), (b) từ `results/baselines_frozen.json` (đóng băng trước NB3); (c) từ `results/verdict.json`.*

**(b) có thật sự mạnh hơn (a) không?** **Có**, và chênh rất xa: 0.000 → 0.765. Prompt naive
khiến model không bao giờ trả về JSON hợp lệ (format = 0). Nhiều khả năng model "suy nghĩ"
hoặc giải thích dài dòng, điều này thể hiện ở latency 3,3 s so với 1,0 s. Vì vậy target của (a)
bằng 0 do **định dạng**, không phải do hiểu sai ticket. Đây cũng là lý do mốc phải so là (b)
chứ không phải (a): so với (a) thì fine-tune "thắng +0.97", một con số vô nghĩa.

**Có sửa `OPTIMIZED_PROMPT` không?** Không. Prompt giữ nguyên bản gốc (SHA khớp).

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6262 | **0.97** | 405.0 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5371 | **0.97** | 272.8 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.00** | 400.5 | 8.78 |
| `qlora` | text-linear, base 4-bit | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.94** | 466.9 | 3.86 |

Cả bốn run đều chạy **30 step**. `attn_only` lệch **0,025%** số tham số so với `correct`
(dưới ngưỡng 5%). Mỗi run đối chứng chỉ đổi một biến so với `correct`: `attn_only` đổi
**vị trí** (rank tăng lên chỉ để giữ ngân sách tham số), `wrong_lr` đổi **LR**, `qlora` đổi
**độ chính xác của base** (16-bit → 4-bit).

**Xếp hạng theo target:** `correct` = `attn_only` (0.97) > `qlora` (0.94) > `wrong_lr` (0.00).
**Xếp hạng theo train loss:** `attn_only` (0.537) > `correct` (0.626) > `qlora` (0.706) > `wrong_lr` (1.570).
Hai thứ tự **không trùng nhau**.

**4.1 — `attn_only` thắng, thua hay hoà?** Trên tập target, `attn_only` **hoà** `correct`:
cả hai đạt 0.97, format 1.0. Theo train loss thì `attn_only` "thắng" rõ (0.537 so với 0.626),
nhưng đó là ảo giác của chỉ số thay thế. `final_loss` ở đây là loss **trung bình cả quá trình**,
và chênh lệch nằm gần như hết ở step 10 (0.825 so với 1.381). Ở step cuối, hai run gần như
bằng nhau (0.02565 so với 0.02562). Vì vậy kết luận trung thực là: **với bài toán này, ở
ngân sách ~32,5M tham số, vị trí gắn adapter không tạo khác biệt đo được trên target.** Thí
nghiệm này *không* xác nhận được luận điểm "vị trí mới là đòn bẩy, không phải rank" của
*LoRA Without Regret*. Nó cũng không bác bỏ luận điểm đó: target đã bão hoà ở 0.97 (chỉ còn
6 trường sai trên 200), nên phép đo không đủ phân giải để tách hai cấu hình. Muốn thấy khác
biệt về vị trí cần một bài khó hơn hoặc ngân sách tham số nhỏ hơn (ví dụ q,v @ r=16 = 1,8M so
với all-linear @ r=16). Một điểm phụ: `attn_only` train nhanh hơn 33% (272,8 s so với 405,0 s)
vì chỉ gắn vào 2 loại module. Trên bài này, đó là cấu hình rẻ hơn với cùng chất lượng.

**4.2 — `wrong_lr` khác gì?** Chỉ đổi LR từ 1e-4 xuống 1e-5 (thang full fine-tune). Đường
loss của `correct` rơi từ 2.163 xuống 0.139 ngay ở epoch 1 rồi xuống ~0.02. Loss của
`wrong_lr` chỉ giảm chậm, đều: 2.163 → 2.066 → 1.606 → 1.326 → 1.141 → 1.119. Nhìn riêng
đường loss này, không có gì "hỏng": nó đi xuống đều đặn, token accuracy tăng từ 0.66 lên 0.79,
và người xem sẽ kết luận *"model đang học, chỉ cần train thêm vài epoch"*. Thực tế target = 0.00
và format = 0.00: sau 30 step, model **chưa học được gì dùng được**, không sinh ra một JSON hợp
lệ nào, và latency 5167 ms cho thấy nó sinh dài tới giới hạn token. Bài học: loss giảm không có
nghĩa là model đạt yêu cầu. Với LoRA, LR phải cao hơn full-FT khoảng 10 lần, và chỉ đánh giá
trên tác vụ thật mới lộ ra lỗi này.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?** Peak VRAM giảm từ 8,78 GB xuống
**3,86 GB** (−4,92 GB, −56%). Cái giá là: target giảm 0.97 → 0.94 (−0.03, tương đương thêm
khoảng 6 trường sai trên 200), train chậm hơn 15% (466,9 s so với 405,0 s, do phải dequantize),
và latency suy luận cao hơn (1725,6 ms so với 1367,1 ms). Số đo của tôi **ủng hộ có điều kiện**
khuyến nghị "không dùng QLoRA cho Qwen3.5": khi model 16-bit đã vừa GPU (như 4B trên T4) thì
QLoRA chỉ thêm chi phí mà không đem lại gì. Tuy vậy, mức tụt 0.03 trên 50 mẫu là nhỏ. Nếu phải
train bản 9B trên T4 thì QLoRA vẫn là lựa chọn hợp lý, chấp nhận đánh đổi một chút chất lượng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**
`target Δ = +0.205` · `regression Δ = −0.224` · `valid_trace_rate = 0.00`

Bản fine-tune **thắng rõ trên tác vụ đích**: 0.765 → 0.970 so với baseline (b) đã prompt tử
tế. Như vậy fine-tune có giá trị thật, chứ không chỉ thắng một prompt yếu. Nhưng nó **trượt
cổng hồi quy**: điểm trên 15 câu hỏi kiến thức/chỉ dẫn phổ thông tụt từ 0.791 xuống 0.567
(−0.224, trong khi ngưỡng cho phép là −0.020). Đây là **quên thảm hoạ** (deck §6.3).

Nguyên nhân hợp lý nhất nằm ở dữ liệu: cả 225 mẫu train đều cùng một dạng (ticket → JSON 4
khoá), không có mẫu phổ thông nào, và model được tối ưu tới loss ~0.02. Trong 30 step với LR
1e-4 trên toàn bộ 12 loại module tuyến tính, adapter đã đẩy phân phối đầu ra về phía "trả lời
ngắn, đúng khuôn JSON". Hệ quả là những câu hỏi mở (dịch, viết câu chúc, kể tên) bị trả lời
thiếu từ khoá. Format = 1.0 và target = 0.97 là bằng chứng adapter học **rất mạnh** một hành
vi duy nhất; regression tụt là mặt trái của cùng hiện tượng đó. Lưu ý: tôi chưa xem từng câu
trả lời regression (results chỉ lưu điểm tổng), nên cơ chế "trả lời theo khuôn" là giả thuyết
cần kiểm chứng.

Ý nghĩa với bài toán: nếu model **chỉ** dùng làm bộ phân loại ticket sau một router, thì
regression không quan trọng và bản fine-tune có thể dùng được. Nếu model đồng thời phục vụ
hội thoại chung, thì **không được deploy** bản này. Hướng sửa theo deck §6.3 là trộn 1–5% dữ
liệu phổ thông (replay) vào tập train, hoặc giảm số step hoặc LR. Tôi không nới ngưỡng cổng
để biến FAILED thành PASSED.

---

## 6. Định tính — bắt buộc có cả ca THUA

Nguồn: `results/qualitative.json` (dự đoán của bản fine-tune trên đủ 50 mẫu, bị cắt ở 100 ký
tự) đối chiếu với nhãn trong `data/eval_target.jsonl`. **Hạn chế:** lab chỉ lưu điểm tổng của
baseline (b), không lưu dự đoán (b) theo từng mẫu, nên cột (b) để trống thay vì đoán. Ở ca
0.75, `intent`/`urgency`/`product` đọc được trực tiếp; `sentiment` bị cắt nhưng phải đúng, vì
điểm 0.75 = 3/4 trường và trường sai đã thấy là `urgency`.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | "…chuột không dây… Cho tôi trả lại. **Gấp.** Shop hỗ trợ tốt." (i=0) | doi_tra · **cao** · chuột không dây · tich_cuc | không lưu | đúng cả 4 (1.00) | ✅ FT đúng: "Gấp" → cao |
| 2 | "…đèn bàn LED… Hoàn tiền. **Quá hạn rồi.** Cảm ơn shop nhiều." (i=2) | hoan_tien · **cao** · đèn bàn LED · tich_cuc | không lưu | đúng cả 4 (1.00) | ✅ FT đúng: hiểu "quá hạn" là khẩn dù không có chữ "gấp" |
| 3 | "…đèn bàn LED… Vỡ khi nhận. Gấp. Shop xem giúp." (i=4) | san_pham_loi · cao · đèn bàn LED · trung_tinh | không lưu | đúng cả 4 (1.00) | ✅ FT đúng: phân biệt trung_tinh với tich_cuc |
| 4 | "…bình giữ nhiệt… Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." (i=3) | hoan_tien · **thap** · bình giữ nhiệt · tich_cuc | không lưu | urgency = **trung_binh** (0.75) | ❌ **FT thua** |
| 5 | "…nồi chiên không dầu… Hoàn tiền. **Khi nào tiện.** Quá tệ." (i=39) | hoan_tien · **thap** · nồi chiên không dầu · tieu_cuc | không lưu | urgency = **trung_binh** (0.75) | ❌ **FT thua** |
| 6 | "…đèn bàn LED… Sai màu. **Khi nào tiện.** Shop hỗ trợ tốt." (i=46) | san_pham_loi · **thap** · đèn bàn LED · tich_cuc | không lưu | urgency = **trung_binh** (0.75) | ❌ **FT thua** |

**Có mẫu chung nào ở các ca FT thua không?** Có, và rất rõ. Bản fine-tune chỉ sai **6/50**
ticket, **cả 6 đều sai đúng một trường** (`urgency`, đoán `trung_binh` thay vì `thap`), và
**cả 6 đều chứa cụm "Khi nào tiện"** (i = 3, 5, 12, 39, 41, 46). Tập eval có đúng 6 ticket
chứa cụm này, tức là model sai **100%** ở cụm này và đúng gần như toàn bộ phần còn lại. Điều
lạ là dữ liệu train có **35 mẫu** chứa "Khi nào tiện", tất cả đều nhãn `thap`, vậy mà model vẫn
không học được. Giả thuyết của tôi: "Khi nào tiện" là tín hiệu ngầm ("bạn rảnh thì làm"),
không có từ khoá "không vội" như các ca `thap` khác, trong khi về mặt ngữ nghĩa câu hỏi "khi
nào" dễ bị model gán là trung bình. 30 step chưa đủ để ghi đè thiên kiến đó. Đây là lỗi
**hệ thống, không phải nhiễu**, nên cách sửa là thêm hoặc nhân trọng số các mẫu chứa cụm này,
hoặc thêm vào prompt một dòng định nghĩa urgency thấp.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi **không deploy** bản fine-tune này như một model đa dụng, dù trên tác vụ
đích nó thắng rõ: target 0.970 so với 0.765 của base đã prompt tốt (+0.205), format tuyệt đối,
và chỉ sai một kiểu lỗi duy nhất. Lý do là cổng hồi quy: khả năng chung tụt 0.224, gấp hơn
mười lần ngưỡng cho phép. Chuỗi nhân quả tôi đọc được từ số liệu như sau. Dữ liệu train đơn
điệu (100% ticket → JSON), cộng với một cấu hình huấn luyện *đúng và mạnh* (all-linear, LR
thang LoRA, loss xuống ~0.02), làm adapter khắc rất sâu một hành vi. Chính sức mạnh đó tạo ra
cả điểm target cao lẫn điểm regression thấp. Bằng chứng là `wrong_lr`, run học yếu nhất, cũng
là run không học được gì hữu ích. Vì vậy đòn bẩy lớn nhất trong lab này **không phải vị trí
adapter hay rank**: `attn_only` và `correct` hoà nhau ở 0.97 với cùng ngân sách tham số. Đòn
bẩy thật nằm ở **learning rate**, là thứ quyết định giữa 0.97 và 0.00, và ở **thành phần dữ
liệu**, là thứ quyết định giữa thắng target và giữ được khả năng chung. Mask đúng là điều kiện
cần chứ không phải đòn bẩy: nó đã được chứng minh đúng ở NB1 (41% token được giám sát), nên
mọi so sánh phía sau mới có nghĩa. Bước tiếp theo hợp lý là train lại `correct` với 1–5% dữ
liệu replay phổ thông và thêm mẫu "Khi nào tiện". Sau đó chạy lại NB5 với **cùng** mốc đã đóng
băng để xem có qua cổng mà không mất phần lớn của +0.205 hay không.

**Ba điều tôi học được** (cụ thể):
1. **Train loss xếp hạng sai.** Theo loss, `attn_only` (0.537) "tốt hơn" `correct` (0.626),
   nhưng theo target thì hai run hoà. Còn `wrong_lr` có đường loss đi xuống rất "khoẻ mạnh"
   nhưng target bằng 0. Từ giờ tôi không chọn checkpoint hay cấu hình bằng loss nữa.
2. **Baseline (a) có thể bằng 0 vì định dạng chứ không vì năng lực.** Prompt naive cho target
   0.000 và format 0.000, nên nếu so với (a), fine-tune nào trông cũng thần kỳ. Một prompt viết
   cẩn thận đã lấy lại 0.765. Phần fine-tune thực sự đóng góp chỉ là +0.205.
3. **Đọc lỗi theo nhóm thay vì theo trung bình.** Con số 0.97 trông như "gần hoàn hảo", nhưng
   khi đối chiếu từng ca thì cả 6 lỗi chung một cụm từ. Lỗi hệ thống như vậy sửa được bằng dữ
   liệu, còn con số trung bình thì che mất nó.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** (1) trộn ~3% mẫu phổ thông vào train để sửa
regression; (2) chạy lại `attn_only` với ngân sách nhỏ (q,v @ r=16, 1,8M tham số) để xem vị
trí có tạo khác biệt khi ngân sách bị siết không; (3) lưu thêm dự đoán của (b) theo từng mẫu
để so trực tiếp ca thắng/thua giữa (b) và (c).

---

## Phụ lục — thưởng đã làm

Không làm phần thưởng trong lần nộp này (chỉ phần core NB1–NB5).

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub

**Ghi chú tái lập:** Colab T4 qua extension Google Colab của VS Code, notebook
`colab/Lab21_RUN_ALL_vscode.ipynb` (bản sao của `Lab21_RUN_ALL.ipynb` với `EVAL_LIMIT=""`,
thêm một ô đóng gói `results/`). Thời gian từng stage: NB1 22 s · NB2 488 s · NB3 471 s ·
NB4 1305 s · NB5 661 s. Một số step log `grad_norm: nan`; đó là lúc fp16 GradScaler bỏ qua
step bị tràn số, và loss vẫn hội tụ bình thường.
