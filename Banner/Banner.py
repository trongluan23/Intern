import openai
import json
import re
import base64
from PIL import Image, ImageDraw, ImageFont, ImageStat
from rembg import remove
import os

openai.api_key = "" 

# ==== Chuyển ảnh sang base64 để gửi GPT vision ====
def image_to_data_url(path):
    with open(path, "rb") as f:
        img_bytes = f.read()
        ext = path.split(".")[-1].lower()
        mime = "jpeg" if ext in ["jpg", "jpeg"] else "png"
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{mime};base64,{b64}"

# ==== Xóa nền ảnh và lưu lại ====
def remove_bg(input_path, output_path):
    with Image.open(input_path) as img:
        out = remove(img)
        out.save(output_path)

# ==== Gọi GPT-4o đọc ảnh và sinh bố cục hợp lý ====
def get_layout_from_images(product_path, logo_path, width=660, height=300):
    product_url = image_to_data_url(product_path)
    logo_url = image_to_data_url(logo_path)

    messages = [
        {"role": "system", "content": "Bạn là nhà thiết kế banner chuyên nghiệp."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
Tôi muốn bạn phân tích ảnh sau để thiết kế banner kích thước {width}x{height}px gồm:
- background
- product
- logo
- title
- description
- website
- cta_button
yêu cầu:

- Tạo bố cục hợp lý, dễ nhìn, làm nổi bật sản phẩm và logo
- áp dụng tỷ lệ thiết kế theo kích cỡ banner
**Ảnh sản phẩm (product) phải to, rõ ràng, không bị cắt bớt ảnh sản phẩm, chiếm nhiều không gian nhất** trong banner  
**Tất cả các thành phần (text, logo, CTA, website...) không được sát mép banner**, mỗi thành phần phải **cách mép tối thiểu 20px** để đảm bảo không bị cắt.
**Nút CTA (cta_button) phải có kích thước không quá lớn,kích thước phụ thuộc vào kích thước banner và tuân theo quy chuẩn phổ biến, dễ nhấn, không quá sát mép banner.**
- Làm nổi bật ảnh sản phẩm so với các thành phần khác  
- Logo đặt ở vị trí 2 góc của banner, không lấn át sản phẩm
- Các thành phần hiển thị đầy đủ, không mất nội dung
- Không thành phần nào được chồng chéo lên nhau
- Mỗi thành phần phải có ít nhất 20px khoảng cách với thành phần khác
- Tối ưu bố cục theo thẩm mỹ, dễ đọc, không mất nội dung
- Trả về JSON duy nhất với mỗi thành phần là [x, y, w, h]. Không có mô tả thêm
"""},
                {"type": "image_url", "image_url": {"url": product_url}},
                {"type": "image_url", "image_url": {"url": logo_url}}
            ]
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=600
    )
    content = response.choices[0].message.content
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group())
    else:
        raise Exception("GPT không trả về JSON hợp lệ")

# ==== Giữ tỷ lệ khi dán ảnh ====
def paste_image_keep_ratio(banner, img_path, box, resample=Image.LANCZOS, cover=False):
    img = Image.open(img_path).convert("RGBA")
    box_x, box_y, box_w, box_h = box
    scale = max(box_w / img.width, box_h / img.height) if cover else min(box_w / img.width, box_h / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img_resized = img.resize((new_w, new_h), resample)

    if cover:
        left, top = (new_w - box_w) // 2, (new_h - box_h) // 2
        img_cropped = img_resized.crop((left, top, left + box_w, top + box_h))
        banner.paste(img_cropped, (box_x, box_y), img_cropped)
    else:
        paste_x = box_x + (box_w - new_w) // 2
        paste_y = box_y + (box_h - new_h) // 2
        banner.paste(img_resized, (paste_x, paste_y), img_resized)

def get_contrasting_text_color(bg_image, box):
    x, y, w, h = box
    region = bg_image.crop((x, y, x + w, y + h))
    stat = ImageStat.Stat(region)
    r, g, b = stat.mean[:3]

    # Tính độ sáng tương đối (perceived brightness)
    brightness = (0.299 * r + 0.587 * g + 0.114 * b)

    # Nếu nền sáng → dùng chữ tối, ngược lại dùng chữ sáng
    return (0, 0, 0) if brightness > 160 else (255, 255, 255)


# ==== Tự động vẽ text vừa box ====
def draw_text_auto_fit_box(draw, text, box, font_path="arial.ttf", max_font_size=40, min_font_size=10, fill_color=(0, 0, 0)):
    x, y, w, h = box
    for size in range(max_font_size, min_font_size - 1, -1):
        try:
            font = ImageFont.truetype(font_path, size)
        except:
            font = ImageFont.load_default()
        lines, line = [], ""
        for word in text.split():
            test_line = f"{line} {word}".strip()
            if draw.textlength(test_line, font=font) <= w:
                line = test_line
            else:
                if line: lines.append(line)
                line = word
        if line: lines.append(line)
        line_h = font.getbbox("Ay")[3] - font.getbbox("Ay")[1]
        total_h = len(lines) * line_h
        if total_h <= h:
            y_offset = y + (h - total_h) // 2
            for l in lines:
                draw.text((x + (w - draw.textlength(l, font=font)) // 2, y_offset), l, font=font, fill=fill_color)
                y_offset += line_h
            return

# ==== Tạo banner chính ====
def compose_banner(bg_path, product_path, logo_path, output_path="banner_final.jpg", width=660, height=300,
                   title="Galaxy S25", description="Camera AI 200MP | Chip Gen 4 | AMOLED 2X",
                   website="www.samsung.com", cta_button="Mua ngay"):
    SCALE = 3  # tạo ảnh với độ phân giải gấp 3 lần
    render_width = width * SCALE
    render_height = height * SCALE

    layout = get_layout_from_images(product_path, logo_path, width, height)
    banner = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    # Background
    if bg_path:
        bg = Image.open(bg_path).convert("RGBA").resize((width, height))
        banner.paste(bg, (0, 0))

    # Product & Logo
    paste_image_keep_ratio(banner, product_path, layout["product"], cover=True)
    paste_image_keep_ratio(banner, logo_path, layout["logo"], cover=False)

    # Text
    draw = ImageDraw.Draw(banner)
    title_color = get_contrasting_text_color(banner, layout["title"])
    desc_color = get_contrasting_text_color(banner, layout["description"])
    website_color = get_contrasting_text_color(banner, layout["website"])

    draw_text_auto_fit_box(draw, title, layout["title"], fill_color=title_color)
    draw_text_auto_fit_box(draw, description, layout["description"], fill_color=desc_color)
    draw_text_auto_fit_box(draw, website, layout["website"], max_font_size=20, min_font_size=12, fill_color=website_color)


    # CTA button
    # if "cta_button" in layout:
    #     x, y, w, h = layout["cta_button"]
    #     draw.rectangle([x, y, x+w, y+h], fill=(255, 0, 0), outline=(0, 0, 0))
    #     draw_text_auto_fit_box(draw, cta_button, layout["cta_button"], max_font_size=28)

    banner_resized = banner.resize((width, height), Image.LANCZOS)
    banner_resized.convert("RGB").save(output_path, format="JPEG", quality=95, dpi=(300, 300))
    print(f"✅ Banner đã tạo tại: {output_path}")

# ==== Demo chạy chính ====
if __name__ == "__main__":
    remove_bg("product2.png", "product2_no_bg.png")
    remove_bg("logo2.png", "logo2_no_bg.png")

    compose_banner(
        bg_path="background3.png",
        product_path="product2_no_bg.png",
        logo_path="logo2_no_bg.png",
        output_path="banner_final.jpg",
        width=300,
        height=250,
        title="Intel Core i5",
        description="13TH Gen | 10-Core | 4.6GHz",
        website="",
        cta_button=""
    )
