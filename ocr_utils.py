"""
Utility OCR untuk Smart NutriScan AI.

Revisi v4 fokus pada kestabilan saat gambar diupload ke Streamlit Cloud.
Perubahan utama:
1. Gambar dibatasi ukurannya agar OCR tidak membebani memori cloud.
2. OCR tidak memakai rotasi otomatis yang berat.
3. EasyOCR membaca numpy array, bukan byte stream, agar lebih kompatibel.
4. Error OCR ditangkap per variasi gambar agar aplikasi tidak langsung crash.
5. Hasil parsing tetap berbasis baris agar nilai gizi dan komposisi lebih logis.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


NUTRITION_DEFAULTS: Dict[str, Any] = {
    "energi": 0.0,
    "lemak_total": 0.0,
    "lemak_jenuh": 0.0,
    "protein": 0.0,
    "karbohidrat": 0.0,
    "gula": 0.0,
    "garam": 0.0,
    "natrium": 0.0,
    "natrium_benzoat": 0.0,
    "komposisi": "Tidak terdeteksi.",
    "product_name": "Produk Tanpa Nama",
}


def normalize_pil_image(pil_image: Image.Image, max_side: int = 1600) -> Image.Image:
    """Membuat gambar aman untuk OCR dan Streamlit Cloud."""
    image = ImageOps.exif_transpose(pil_image).convert("RGB")
    width, height = image.size
    longest = max(width, height)

    if longest > max_side:
        ratio = max_side / float(longest)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def preprocess_image_variants_for_ocr(pil_image: Image.Image) -> Dict[str, Image.Image]:
    """Menghasilkan variasi gambar ringan agar OCR tidak bergantung pada satu metode preprocessing."""
    safe_image = normalize_pil_image(pil_image)
    img = np.array(safe_image)

    # Perbesar secukupnya. Jangan terlalu besar agar tidak boros memori cloud.
    h, w = img.shape[:2]
    if max(h, w) < 1200:
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=8,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # Binary kadang bagus, kadang merusak teks kecil. Tetap disiapkan sebagai variasi ringan.
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return {
        "original_resized": Image.fromarray(img),
        "gray_enhanced": Image.fromarray(denoised),
        "sharpened": Image.fromarray(sharpened),
        "binary": Image.fromarray(binary),
    }


def _bbox_center_y(item: Dict[str, Any]) -> float:
    return float(sum(point[1] for point in item["bbox"]) / 4)


def _bbox_left_x(item: Dict[str, Any]) -> float:
    return float(min(point[0] for point in item["bbox"]))


def deduplicate_ocr_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Menghapus teks duplikat dari beberapa hasil preprocessing, memilih confidence tertinggi."""
    unique: Dict[str, Dict[str, Any]] = {}

    for item in results:
        key = re.sub(r"\s+", " ", item["text"].lower().strip())
        if not key:
            continue

        if key not in unique or item["conf"] > unique[key]["conf"]:
            unique[key] = item

    return list(unique.values())


def _reader_readtext_safely(reader: Any, image_variant: Image.Image) -> Tuple[List[Any], str]:
    """Menjalankan EasyOCR dengan parameter aman dan fallback jika versi EasyOCR berbeda."""
    image_array = np.array(image_variant)

    try:
        results = reader.readtext(
            image_array,
            detail=1,
            paragraph=False,
            decoder="greedy",
            contrast_ths=0.1,
            adjust_contrast=0.6,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=1.0,
        )
        return results, ""
    except TypeError as exc:
        # Fallback untuk EasyOCR yang tidak menerima salah satu parameter tambahan.
        try:
            results = reader.readtext(image_array, detail=1, paragraph=False)
            return results, ""
        except Exception as inner_exc:
            return [], f"EasyOCR fallback gagal: {inner_exc}"
    except Exception as exc:
        return [], f"EasyOCR gagal membaca gambar: {exc}"


def run_ocr_multi_variant(
    reader: Any,
    pil_image: Image.Image,
    min_confidence: float = 0.20,
    max_variants: int = 3,
) -> Dict[str, Any]:
    """Menjalankan EasyOCR pada beberapa variasi gambar tanpa membuat aplikasi crash."""
    variants = preprocess_image_variants_for_ocr(pil_image)
    selected_variants = dict(list(variants.items())[:max_variants])

    all_results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for variant_name, image_variant in selected_variants.items():
        results, error_message = _reader_readtext_safely(reader, image_variant)
        if error_message:
            errors.append(f"{variant_name}: {error_message}")
            continue

        for result in results:
            try:
                bbox, text, conf = result
            except Exception:
                continue

            clean_text = str(text).strip()
            try:
                confidence = float(conf)
            except Exception:
                confidence = 0.0

            if clean_text and confidence >= min_confidence:
                all_results.append({
                    "variant": variant_name,
                    "bbox": bbox,
                    "text": clean_text,
                    "conf": confidence,
                })

    deduped = deduplicate_ocr_results(all_results)
    lines = group_ocr_results_into_lines(deduped)

    return {
        "items": deduped,
        "lines": lines,
        "raw_text": "\n".join(lines),
        "variants": selected_variants,
        "errors": errors,
    }


def group_ocr_results_into_lines(ocr_items: List[Dict[str, Any]], y_tolerance: int = 18) -> List[str]:
    """Menyusun ulang hasil OCR menjadi baris berdasarkan posisi vertikal dan horizontal."""
    if not ocr_items:
        return []

    sorted_items = sorted(ocr_items, key=lambda item: (_bbox_center_y(item), _bbox_left_x(item)))
    lines: List[Dict[str, Any]] = []

    for item in sorted_items:
        y_center = _bbox_center_y(item)
        placed = False

        for line in lines:
            if abs(line["y"] - y_center) <= y_tolerance:
                line["items"].append(item)
                line["y"] = (line["y"] + y_center) / 2
                placed = True
                break

        if not placed:
            lines.append({"y": y_center, "items": [item]})

    text_lines: List[str] = []
    for line in sorted(lines, key=lambda value: value["y"]):
        ordered_items = sorted(line["items"], key=_bbox_left_x)
        line_text = " ".join(item["text"] for item in ordered_items)
        line_text = re.sub(r"\s+", " ", line_text).strip()
        if line_text:
            text_lines.append(line_text)

    return text_lines


def normalize_ocr_text(text: str) -> str:
    """Normalisasi kata OCR umum pada label Indonesia dan Inggris."""
    text = str(text).lower().strip()
    text = text.replace(",", ".")

    replacements = {
        "nutrition facts": "informasi nilai gizi",
        "nutrition fact": "informasi nilai gizi",
        "energy": "energi",
        "calories": "energi",
        "calorie": "energi",
        "kalori": "energi",
        "energi total": "energi",
        "jumlah energi": "energi",
        "total fat": "lemak total",
        "lemak tota1": "lemak total",
        "lemak totai": "lemak total",
        "saturated fat": "lemak jenuh",
        "total carbohydrate": "karbohidrat",
        "carbohydrate": "karbohidrat",
        "karbohidrat total": "karbohidrat",
        "sugars": "gula",
        "sugar": "gula",
        "sodium": "natrium",
        "salt": "garam",
        "ingredients": "komposisi",
        "ingredient": "komposisi",
        "bahan bahan": "komposisi",
        "bahan-bahan": "komposisi",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"(?<=\d)(kkal|kal|mg|g)\b", r" \1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_numbers_with_units(line: str) -> List[Tuple[float, str]]:
    clean_line = normalize_ocr_text(line)
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(kkal|kal|mg|g)?", clean_line)
    values: List[Tuple[float, str]] = []

    for number, unit in matches:
        try:
            values.append((float(number), unit or ""))
        except ValueError:
            continue

    return values


def extract_number_from_line(line: str, preferred_units: Tuple[str, ...] | None = None) -> float:
    """Mengambil angka paling logis dari satu baris label."""
    values = _extract_numbers_with_units(line)
    if not values:
        return 0.0

    if preferred_units:
        for value, unit in values:
            if unit in preferred_units:
                return float(value)

    return float(values[0][0])


def parse_nutrition_from_lines(lines: List[str]) -> Dict[str, Any]:
    """Mengekstrak nilai gizi dari baris teks OCR."""
    data = dict(NUTRITION_DEFAULTS)

    for line in lines:
        clean_line = normalize_ocr_text(line)
        if not clean_line:
            continue

        if "energi" in clean_line:
            data["energi"] = extract_number_from_line(clean_line, preferred_units=("kkal", "kal"))
        elif "lemak jenuh" in clean_line:
            data["lemak_jenuh"] = extract_number_from_line(clean_line, preferred_units=("g",))
        elif "lemak total" in clean_line or clean_line.startswith("lemak"):
            data["lemak_total"] = extract_number_from_line(clean_line, preferred_units=("g",))
        elif "protein" in clean_line:
            data["protein"] = extract_number_from_line(clean_line, preferred_units=("g",))
        elif "karbohidrat" in clean_line:
            data["karbohidrat"] = extract_number_from_line(clean_line, preferred_units=("g",))
        elif "gula" in clean_line:
            data["gula"] = extract_number_from_line(clean_line, preferred_units=("g",))
        elif "natrium benzoat" in clean_line or "benzoat" in clean_line:
            data["natrium_benzoat"] = extract_number_from_line(clean_line, preferred_units=("mg", "g"))
        elif "natrium" in clean_line:
            data["natrium"] = extract_number_from_line(clean_line, preferred_units=("mg",))
        elif "garam" in clean_line:
            data["garam"] = extract_number_from_line(clean_line, preferred_units=("g", "mg"))

    if data["garam"] > 0 and data["natrium"] == 0:
        data["natrium"] = data["garam"] * 400

    product_name = infer_product_name(lines)
    if product_name:
        data["product_name"] = product_name

    return data


def infer_product_name(lines: List[str]) -> str:
    """Mengambil kandidat nama produk dari baris awal, tanpa memaksa jika yang terbaca adalah label gizi."""
    blocked = [
        "informasi nilai gizi",
        "nutrition facts",
        "komposisi",
        "energi",
        "lemak",
        "protein",
        "karbohidrat",
        "gula",
        "natrium",
        "takaran",
    ]

    for line in lines[:5]:
        clean = normalize_ocr_text(line)
        if len(clean) < 3:
            continue
        if any(word in clean for word in blocked):
            continue
        if re.search(r"\d", clean):
            continue
        return str(line).strip().title()

    return "Produk Tanpa Nama"


def parse_composition_from_lines(lines: List[str]) -> str:
    """Mengekstrak komposisi dari hasil OCR."""
    raw_text = " ".join(lines).strip()
    if not raw_text:
        return "Tidak terdeteksi."

    normalized = normalize_ocr_text(raw_text)
    match = re.search(r"(?:komposisi|bahan)\s*:?\s*(.*)", normalized, flags=re.IGNORECASE)
    composition = match.group(1) if match else normalized

    stop_patterns = [
        "informasi nilai gizi",
        "nutrition facts",
        "mengandung alergen",
        "diproduksi",
        "baik digunakan",
        "expired",
        "exp",
        "tanggal",
        "berat bersih",
    ]

    for stop_word in stop_patterns:
        composition = re.split(stop_word, composition, flags=re.IGNORECASE)[0]

    composition = re.sub(r"\s+", " ", composition).strip(" .,:;")
    if len(composition) < 5:
        return "Tidak terdeteksi."

    return composition.capitalize()


def parse_scan_result(reader: Any, pil_image: Image.Image, mode: str = "nutrition") -> Dict[str, Any]:
    """Fungsi praktis untuk dipakai di app.py."""
    ocr_payload = run_ocr_multi_variant(reader, pil_image)
    lines = ocr_payload["lines"]

    if mode == "composition":
        parsed = dict(NUTRITION_DEFAULTS)
        parsed["komposisi"] = parse_composition_from_lines(lines)
    else:
        parsed = parse_nutrition_from_lines(lines)

    return {
        "parsed": parsed,
        "lines": lines,
        "raw_text": ocr_payload["raw_text"],
        "items": ocr_payload["items"],
        "variants": ocr_payload["variants"],
        "errors": ocr_payload.get("errors", []),
    }
