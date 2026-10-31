# CapCut AI Colorist & Draft Tech

**Bộ đôi Skill chỉnh màu phim 35mm điện ảnh cho project CapCut macOS — đo thực nghiệm từng đoạn cắt (`source_timerange`), bám sát phân cảnh Bảng Tra/Kịch bản, chống nhảy màu khi đổi góc máy và tinh chỉnh đủ 12 thanh trượt Adjust cho TỪNG VIDEO CON (`child video segment`).**

[English](README.en.md) · [AI Colorist Guide](skills/capcut-ai-colorist/SKILL.md) · [Draft Tech Engine](skills/capcut-draft-tech/SKILL.md) · [MIT](LICENSE)

Repo tách biệt rõ **2 tầng skill**:
1. **[`capcut-ai-colorist`](skills/capcut-ai-colorist/SKILL.md)** *(Hướng dẫn tư duy & chiến lược màu cho AI)*:
   - **Bắt buộc AI Trực Tiếp Xem Ảnh/Frame Bằng Mắt (`read_file` / `view_file` Trước & Sau Khi Chỉnh) + Hỏi User Trước Khi Áp Màu**:
     - Khi chạy `analyze_project.py`, tool tự động xuất bảng ảnh gốc của toàn bộ phân cảnh (`/tmp/<project>_raw_scenes.jpg`) — **AI bắt buộc mở xem trực tiếp bảng ảnh gốc này** rồi mới hỏi User nguồn model (Seedance 2.5, Google Omni 1.1, Grok, hay hỗn hợp) và Mood cảm xúc (`narrative-arc`, `warm-human`, `dramatic-contrast`, `luxury-crisp`, `vintage-35mm`).
     - Sau khi chạy `apply_grade.py`, tool tự động kết xuất bảng đối chiếu trực quan **`RAW vs. GRADED` (`/tmp/<project>_graded_comparison.jpg`)** áp trực tiếp bảng 3D `.cube` + lớp bảo vệ da + 12 thanh trượt Adjust lên frame thật của từng phân cảnh — **AI bắt buộc mở xem ảnh đối chiếu `RAW vs. GRADED` bằng mắt** để kiểm tra da người, độ tương phản và sự đồng bộ giữa các cảnh; nếu cảnh nào lệch mắt sẽ ghi đè ngay qua `--plan` và chạy lại!
   - **Phân cảnh chuẩn 100% từ Bảng Tra / Kịch bản gốc (`*BANGTRA*.txt`, `*PROMPT*.txt`, `sf-board.json`)**: Tự động gộp các góc máy đảo chiều trong cùng 1 phòng (`BG1a` + `BG1b` → `BG1`) để không bao giờ đoán nhầm phân cảnh.
   - **Khóa Đích Màu Của Cảnh + LUT Bù Trừ Theo Góc Máy (`Angle-Adaptive Sibling LUT`)**: Trong shader của CapCut, LUT chạy trước (`zorder: 2000`) và Adjust chạy sau (`zorder: 8004`). Các góc máy bình thường dùng chung Primary Scene LUT; các góc nhảy sáng/nhảy nhiệt độ màu gắt tự động chuyển sang LUT anh em bù trừ ở tầng `zorder: 2000` + kéo thanh trượt Adjust ở tầng `zorder: 8004` để mọi góc máy hội tụ về cùng 1 màu cuối.
   - **Adjust riêng biệt cho TỪNG VIDEO CON (`child video segment`)**: Không bao giờ chỉ chỉnh lớp vỏ Compound Clip ngoài cùng — mỗi video con bên trong Compound Clip lẫn ngoài Timeline đều được đo theo đúng `source_timerange` và gắn riêng **11–12 thông số Adjust** (`shadow`, `highlight`, `brightness`, `light_sensation`, `contrast`, `tone`, `temperature`, `saturation`, `sharpen`, `clear`, `particle`, `vignetting`) + `lut` (`enable_skin_tone_correction: true`).
2. **[`capcut-draft-tech`](skills/capcut-draft-tech/SKILL.md)** *(Công cụ kỹ thuật & JSON Engine)*:
   - Đóng gói sẵn **trọn bộ 1,636 LUT (`.cube` / `.3dl`)** trực tiếp trong [`skills/capcut-draft-tech/resources/luts/`](skills/capcut-draft-tech/resources/luts) kèm cơ sở dữ liệu đo quang học 3D của toàn bộ **1,636 LUT** tại [`lut_optical_db.json`](skills/capcut-draft-tech/resources/lut_optical_db.json).
   - **Check Màu Video Thực Tế Trước Khi Chọn LUT (`diagnose_video_state`) + Giả Lập Toán Học `Video × 1,636 LUT` (`simulate_video_through_lut`)**:
     - Thay vì khóa cứng máy móc theo tên file, trước khi chọn bất kỳ LUT nào, thuật toán **đo trạng thái quang học thực tế của chính video đó** (`FLAT_LOG_OR_FOGGY` nếu video bị phẳng/mù sương như Log, `HIGH_KEY_MILKY` nếu dư sáng trắng bệch, `LOW_KEY_DARK` nếu sập tối, `HARSH_CONTRAST` nếu gắt sáng, `HEAVY_WARM_CAST` nếu ám cam, `COLD_CYAN_CAST` nếu ám lạnh).
     - Tiếp theo, thuật toán **nhân thử (`simulate`) thông số thật của video qua từng LUT trong cả 1,636 LUT** ở các mức cường độ `alpha` (`18% .. 32%`) kết hợp mô hình bảo vệ da AI `tt_skin_seg` (`"enable_skin_tone_correction": true`) của CapCut để tính ra màu đầu ra dự kiến (`sim_p5, sim_mid, sim_hi, sim_rb, sim_gm, sim_sat`).
     - Nếu một video thực tế bị phẳng/mù (`FLAT_LOG_OR_FOGGY`), các LUT kéo tương phản mạnh / Log-recovery sẽ tự động được mở khóa vì kết quả giả lập `sim_p5, sim_hi` trên video đó rơi đúng vào chuẩn 35mm; ngược lại, trên video đã nét/đậm sẵn, những LUT làm cháy sáng (`sim_hi > 0.85`) hoặc bết đen (`sim_p5 < 0.014`) trên chính video đó sẽ tự động bị loại!
   - Đo 16 chỉ số quang học/màu sắc bằng `ffmpeg` + `numpy` cho từng đoạn cắt (`source_timerange`).
   - Đồng bộ an toàn toàn bộ `draft_info.json`, `draft_info.json.bak`, `template-2.tmp`, `Timelines/*/draft_info.json`, `subdraft/*/draft_content.json` và xoá `patch.json`.

---

## Cài đặt trên máy

Cần **Python 3.10+**, `numpy`, `ffmpeg` và Git:

```sh
git clone https://github.com/Ductoii/capcut-color-grading.git
cd capcut-color-grading
python3 install.py
```

Mặc định `install.py` cài cả 2 skill (`capcut-ai-colorist` và `capcut-draft-tech`) vào `~/.agents/skills/` và tự động tải + nạp trọn bộ **1,636 LUT `.cube` / `.3dl`** vào CapCut (`~/Movies/CapCut/User Data/Resources/Lut/LutImportCfg.json`).

Nếu dùng thư mục skills khác (ví dụ `~/.gemini/config/skills` hoặc `~/.codex/skills`):

```sh
python3 install.py --dest ~/.gemini/config/skills
```

---

## Cách gọi trong Codex / Antigravity

```text
$capcut-ai-colorist
Chỉnh màu chuẩn phim 35mm cho project CapCut "1030" (hoặc tên project của bạn).
Đo thông số thực tế từng đoạn cắt, đối chiếu Bảng Tra kịch bản, hỏi mình nguồn video + cảm xúc mong muốn trước khi áp màu, và nhớ adjust đầy đủ cho từng video con trong clip ghép.
```

Hoặc chạy trực tiếp bằng CLI kỹ thuật sau khi đã chốt hướng màu:

```sh
# 1. Đo thông số từng video con (source_timerange) & dò phân cảnh từ Bảng Tra
python3 ~/.agents/skills/capcut-draft-tech/scripts/analyze_project.py "1030" --out /tmp/1030_metrics.json

# 2. Áp LUT + 12 thanh trượt Adjust cho từng video con & đồng bộ subdraft CapCut
python3 ~/.agents/skills/capcut-draft-tech/scripts/apply_grade.py "1030" \
  --metrics /tmp/1030_metrics.json \
  --emotion narrative-arc \
  --auto-match-outliers \
  --relaunch
```
