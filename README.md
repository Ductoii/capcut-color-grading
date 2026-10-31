# CapCut AI Colorist & Draft Tech

**Bộ đôi Skill chỉnh màu phim 35mm điện ảnh cho project CapCut macOS — đo thực nghiệm từng đoạn cắt (`source_timerange`), bám sát phân cảnh Bảng Tra/Kịch bản, chống nhảy màu khi đổi góc máy và tinh chỉnh đủ 12 thanh trượt Adjust cho TỪNG VIDEO CON (`child video segment`).**

[English](README.en.md) · [AI Colorist Guide](skills/capcut-ai-colorist/SKILL.md) · [Draft Tech Engine](skills/capcut-draft-tech/SKILL.md) · [MIT](LICENSE)

Repo tách biệt rõ **2 tầng skill**:
1. **[`capcut-ai-colorist`](skills/capcut-ai-colorist/SKILL.md)** *(Hướng dẫn tư duy & chiến lược màu cho AI)*:
   - **Bắt buộc hỏi User trước khi chỉnh (Mandatory Pre-Grading Intake)**: Hỏi nguồn model (Seedance 2.5, Google Omni 1.1, Grok, hay hỗn hợp) và Mood/Hướng cảm xúc (`narrative-arc`, `warm-human`, `dramatic-contrast`, `luxury-crisp`, `vintage-35mm`) sau khi đo thông số thực tế.
   - **Phân cảnh chuẩn 100% từ Bảng Tra / Kịch bản gốc (`*BANGTRA*.txt`, `*PROMPT*.txt`, `sf-board.json`)**: Tự động gộp các góc máy đảo chiều trong cùng 1 phòng (`BG1a` + `BG1b` → `BG1`) để không bao giờ đoán nhầm phân cảnh.
   - **Khóa Đích Màu Của Cảnh + LUT Bù Trừ Theo Góc Máy (`Angle-Adaptive Sibling LUT`)**: Trong shader của CapCut, LUT chạy trước (`zorder: 2000`) và Adjust chạy sau (`zorder: 8004`). Các góc máy bình thường dùng chung Primary Scene LUT; các góc nhảy sáng/nhảy nhiệt độ màu gắt tự động chuyển sang LUT anh em bù trừ ở tầng `zorder: 2000` + kéo thanh trượt Adjust ở tầng `zorder: 8004` để mọi góc máy hội tụ về cùng 1 màu cuối.
   - **Adjust riêng biệt cho TỪNG VIDEO CON (`child video segment`)**: Không bao giờ chỉ chỉnh lớp vỏ Compound Clip ngoài cùng — mỗi video con bên trong Compound Clip lẫn ngoài Timeline đều được đo theo đúng `source_timerange` và gắn riêng **11–12 thông số Adjust** (`shadow`, `highlight`, `brightness`, `light_sensation`, `contrast`, `tone`, `temperature`, `saturation`, `sharpen`, `clear`, `particle`, `vignetting`) + `lut` (`enable_skin_tone_correction: true`).
2. **[`capcut-draft-tech`](skills/capcut-draft-tech/SKILL.md)** *(Công cụ kỹ thuật & JSON Engine)*:
   - Bundled sẵn **12 LUT `.cube` 35mm cốt lõi** trong `skills/capcut-draft-tech/resources/luts/` và tự động đăng ký vào `LutImportCfg.json` của CapCut.
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

Mặc định `install.py` cài cả 2 skill (`capcut-ai-colorist` và `capcut-draft-tech`) vào `~/.agents/skills/` và tự động nạp 12 LUT `.cube` cốt lõi vào CapCut (`~/Movies/CapCut/User Data/Resources/Lut/LutImportCfg.json`).

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
