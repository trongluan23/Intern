import openai
import json
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai


# === Cấu hình Gemini API Key ===
genai.configure(api_key="")  # Thay bằng API key của bạn

# === Input dữ liệu ===
input_data = {
    "background_image": "background.jpg",
    "product_image": "product.jpg",
    "logo_image": "logo.png",
    "company_name": "Samsung",
    "product_name": "Galaxy S25",
    "description": "Camera AI 200MP | Chip Snapdragon Gen 4 | Màn hình Dynamic AMOLED 2X",
    "website": "www.samsung.com",
    "phone": "1800-588-889",
    "banner_size": [300, 600]
}


# === Bước 1: Sinh layout từ Gemini ===
def get_layout_from_gemini(input_data):
    prompt = f"""
Bạn là một nhà thiết kế banner. Hãy tạo bố cục cho banner kích thước {input_data['banner_size'][0]}x{input_data['banner_size'][1]} px.

Các thành phần:
- Ảnh nền: full nền
- Logo: {input_data['logo_image']}
- Ảnh sản phẩm: {input_data['product_image']}
- Tên sản phẩm: {input_data['product_name']}
- Mô tả: {input_data['description']}
- Website: {input_data['website']}
- SĐT: {input_data['phone']}

Trả về JSON list định dạng:
[
  {{ "type": "image", "src": "background", "x": 0, "y": 0, "width": W, "height": H }},
  {{ "type": "image", "src": "logo", "x": ..., "y": ..., "width": ..., "height": ... }},
  {{ "type": "text", "text": "Galaxy S25", "x": ..., "y": ..., "font_size": ..., "color": "#FFFFFF" }},
  ...
]
Chỉ cần nội dung JSON.
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    layout_json = response.text.strip()
    print("DEBUG Gemini output:\n", layout_json)  # In ra để kiểm tra

    # Loại bỏ code block markdown nếu có
    if layout_json.startswith("```"):
        layout_json = layout_json.split('\n', 1)[1]  # Bỏ dòng đầu
        if layout_json.endswith("```"):
            layout_json = layout_json.rsplit('\n', 1)[0]  # Bỏ dòng cuối

    layout_json = layout_json.strip()

    # Kiểm tra nếu kết quả rỗng hoặc không phải JSON
    if not layout_json or not layout_json.startswith("["):
        raise ValueError("Gemini không trả về JSON hợp lệ. Nội dung trả về:\n" + layout_json)

    return json.loads(layout_json)


# === Bước 2: Vẽ banner theo layout ===
def draw_banner(layout, input_data):
    # Load ảnh
    bg = Image.open(input_data["background_image"]).resize(tuple(input_data["banner_size"]))
    banner = bg.copy()
    draw = ImageDraw.Draw(banner)

    # Load các ảnh phụ
    logo = Image.open(input_data["logo_image"]).convert("RGBA")
    product = Image.open(input_data["product_image"]).convert("RGBA")

    # Load font (nên có arial.ttf hoặc Roboto)
    try:
        font_path = "arial.ttf"
        default_font = ImageFont.truetype(font_path, 20)
    except:
        default_font = ImageFont.load_default()

    for item in layout:
        if item["type"] == "image":
            if item["src"] == "logo":
                img = logo.resize((item["width"], item["height"]))
                banner.paste(img, (item["x"], item["y"]), img)
            elif item["src"] == "product":
                img = product.resize((item["width"], item["height"]))
                banner.paste(img, (item["x"], item["y"]), img)

        elif item["type"] == "text":
            font_size = item.get("font_size", 16)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except:
                font = default_font
            draw.text((item["x"], item["y"]), item["text"], font=font, fill=item.get("color", "#FFFFFF"))

    # Lưu kết quả
    banner.save("output_banner.png")
    print("✅ Banner đã được tạo tại: output_banner.png")


# === Thực thi ===
if __name__ == "__main__":
    layout = get_layout_from_gemini(input_data)
    draw_banner(layout, input_data)
