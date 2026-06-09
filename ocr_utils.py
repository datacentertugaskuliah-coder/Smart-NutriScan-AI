"""
Utility OCR untuk Smart NutriScan AI.
Fokus file ini adalah membaca label nilai gizi dan komposisi produk dengan lebih stabil.
Alur utama:
1. Membuat beberapa variasi gambar.
2. Menjalankan EasyOCR dengan detail posisi teks.
3. Mengelompokkan hasil OCR berdasarkan baris.
4. Mengekstrak nilai gizi dan komposisi dari baris teks.
"""

from __future__ import annotations

import io
import re
from typing import Dict, List, Any

import cv2
import numpy as np
from PIL import Image


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


def preprocess_image_variants_for_ocr(pil_image: Image.Image) -> Dict[str, Image.Image]:
    """Menghasilkan beberapa versi gambar agar OCR tidak bergantung pada satu metode preprocessing."""
    img = np.array(pil_image.convert("RGB"))

    img_big = cv2.resize(
        img,
        None,
        fx=2.0,
        fy=2.0,
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(img_big, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    denoised = cv2.fastNlMeansDenoising(
        enhanced,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    return {
        "original_resized": Image.fromarray(img_big),
        "gray_enhanced": Image.fromarray(denoised),
        "binary": Image.fromarray(binary),
        "sharpened": Image.fromarray(sharpened),
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


def run_ocr_multi_variant(reader: Any, pil_image: Image.Image, min_confidence: float = 0.20) -> Dict[str, Any]:
    """Menjalankan EasyOCR pada beberapa variasi gambar dan mengembalikan item teks terstruktur."""
    variants = preprocess_image_variants_for_ocr(pil_image)
    all_results: List[Dict[str, Any]] = []

    for variant_name, image_variant in variants.items():
        img_byte_arr = io.BytesIO()
        image_variant.save(img_byte_arr, format="PNG")

        results = reader.readtext(
            img_byte_arr.getvalue(),
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            beamWidth=5,
            contrast_ths=0.05,
            adjust_contrast=0.7,
            text_threshold=0.5,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=2,
            rotation_info=[90, 180, 270],
        )

        for bbox, text, conf in results:
            clean_text = str(text).strip()
            if clean_text and float(conf) >= min_confidence:
                all_results.append({
                    "variant": variant_name,
                    "bbox": bbox,
                    "text": clean_text,
                    "conf": float(conf),
                })

    deduped = deduplicate_ocr_results(all_results)
    lines = group_ocr_results_into_lines(deduped)

    return {
        "items": deduped,
        "lines": lines,
        "raw_text": "\n".join(lines),
        "variants": variants,
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
        "lemak jenuh": "lemak jenuh",
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


def _extract_numbers_with_units(line: str) -> List[tuple[float, str]]:
    clean_line = normalize_ocr_text(line)
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(kkal|kal|mg|g)?", clean_line)
    values: List[tuple[float, str]] = []

    for number, unit in matches:
        try:
            values.append((float(number), unit or ""))
        except ValueError:
            continue

    return values


def extract_number_from_line(line: str, preferred_units: tuple[str, ...] | None = None) -> float:
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

    match = re.search(r"(?:komposisi|bahan)\s*:?[\s]*(.*)", normalized, flags=re.IGNORECASE)
    if match:
        composition = match.group(1)
    else:
        composition = normalized

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
    }
