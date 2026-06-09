"""
Utility OCR untuk Smart NutriScan AI.

Revisi v8 fokus pada akurasi logis hasil OCR dan koreksi sisa kasus gula 1 g yang terbaca 19.
Masalah yang diperbaiki:
1. Hasil dari beberapa variasi preprocessing tidak lagi digabung mentah karena bisa menduplikasi teks.
2. Parsing nilai gizi memakai kandidat per variasi gambar, lalu memilih nilai yang paling masuk akal.
3. Persentase AKG tidak lagi ikut terbaca sebagai nilai gram atau mg.
4. Nilai tidak wajar seperti 259 g untuk lemak jenuh diperbaiki dengan aturan desimal yang ketat.
5. Takaran saji ikut dibaca dari label.
6. Komposisi dibaca dari satu variasi terbaik agar tidak berulang dua sampai tiga kali.
7. Huruf g yang sering terbaca sebagai angka 9 diperbaiki sebelum angka masuk ke form.
8. Gula divalidasi terhadap karbohidrat total agar 1g yang terbaca 19 tidak masuk sebagai 19 g.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


NUTRITION_DEFAULTS: Dict[str, Any] = {
    "takaran_saji": 100.0,
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

NUTRITION_FIELD_LIMITS: Dict[str, Tuple[float, float]] = {
    "takaran_saji": (1.0, 1000.0),
    "energi": (0.0, 1000.0),
    "lemak_total": (0.0, 80.0),
    "lemak_jenuh": (0.0, 40.0),
    "protein": (0.0, 80.0),
    "karbohidrat": (0.0, 150.0),
    "gula": (0.0, 120.0),
    "garam": (0.0, 20.0),
    "natrium": (0.0, 5000.0),
    "natrium_benzoat": (0.0, 2000.0),
}


def normalize_pil_image(pil_image: Image.Image, max_side: int = 1100) -> Image.Image:
    """Membuat gambar aman untuk OCR dan menjaga orientasi EXIF."""
    image = ImageOps.exif_transpose(pil_image).convert("RGB")
    width, height = image.size
    longest = max(width, height)

    if longest > max_side:
        ratio = max_side / float(longest)
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        image = image.resize(new_size, Image.LANCZOS)

    return image


def preprocess_image_variants_for_ocr(pil_image: Image.Image) -> Dict[str, Image.Image]:
    """Menghasilkan variasi gambar. Tidak semua variasi akan dipakai sebagai sumber akhir."""
    safe_image = normalize_pil_image(pil_image)
    img = np.array(safe_image)

    h, w = img.shape[:2]
    if max(h, w) < 900:
        img = cv2.resize(img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)

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

    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Urutan sengaja dibuat dari yang paling natural ke yang paling agresif.
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


def group_ocr_results_into_lines(ocr_items: List[Dict[str, Any]], y_tolerance: int = 18) -> List[str]:
    """Menyusun hasil OCR per variasi menjadi baris, tidak lintas variasi."""
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


def _reader_readtext_safely(reader: Any, image_variant: Image.Image) -> Tuple[List[Any], str]:
    """Menjalankan EasyOCR dengan parameter aman dan fallback."""
    image_array = np.array(image_variant)

    try:
        results = reader.readtext(
            image_array,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
            beamWidth=3,
            contrast_ths=0.08,
            adjust_contrast=0.7,
            text_threshold=0.45,
            low_text=0.25,
            link_threshold=0.35,
            mag_ratio=1.1,
        )
        return results, ""
    except TypeError:
        try:
            results = reader.readtext(image_array, detail=1, paragraph=False)
            return results, ""
        except Exception as inner_exc:
            return [], f"EasyOCR fallback gagal: {inner_exc}"
    except Exception as exc:
        return [], f"EasyOCR gagal membaca gambar: {exc}"


def normalize_ocr_text(text: str) -> str:
    """Normalisasi kata OCR umum pada label Indonesia dan Inggris."""
    text = str(text).lower().strip()
    text = text.replace("\n", " ")
    text = text.replace(",", ".")
    text = re.sub(r"[|]", " ", text)

    # Koreksi karakter OCR yang sering keliru pada angka dan satuan.
    text = re.sub(r"\b[oO]\s*(g|mg|kkal|kal)\b", r"0 \1", text)
    text = re.sub(r"(?<=\d)[oO](?=\d|\s*(?:g|mg|kkal|kal)\b)", "0", text)
    text = re.sub(r"(?<=\d)l(?=\d)", "1", text)

    replacements = {
        "nutrition information": "informasi nilai gizi",
        "nutrition facts": "informasi nilai gizi",
        "nutrition fact": "informasi nilai gizi",
        "serving size": "takaran saji",
        "serving amount": "jumlah per sajian",
        "amount per serving": "jumlah per sajian",
        "energy total": "energi total",
        "total calories": "energi total",
        "calories": "energi",
        "calorie": "energi",
        "energy from fat": "energi dari lemak",
        "kalori": "energi",
        "total fat": "lemak total",
        "lemak tota1": "lemak total",
        "lemak totai": "lemak total",
        "saturated fat": "lemak jenuh",
        "lemak jenu h": "lemak jenuh",
        "lemak jen uh": "lemak jenuh",
        "protein/protein": "protein",
        "total carbohydrate": "karbohidrat total",
        "carbohydrate": "karbohidrat",
        "karbohidrat tota1": "karbohidrat total",
        "karbohidrat totai": "karbohidrat total",
        "sugars": "gula",
        "sugar": "gula",
        "sodium": "natrium",
        "salt": "garam",
        "ingredients": "komposisi",
        "ingredient": "komposisi",
        "bahan bahan": "komposisi",
        "bahan-bahan": "komposisi",
        "ingredienta": "ingredienta",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"(?<=\d)(kkal|kal|mg|g)\b", r" \1", text)
    text = re.sub(r"(?<=\d)\s*%", " %", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



GRAM_BASED_FIELDS = {
    "lemak_total",
    "lemak_jenuh",
    "protein",
    "karbohidrat",
    "gula",
    "garam",
}


def repair_gram_unit_read_as_nine(line: str, field: str) -> str:
    """Memperbaiki pola OCR saat huruf g terbaca sebagai angka 9.

    Contoh nyata pada label kecil:
    5g  -> 59
    2.5g -> 2.59 atau 259
    14g -> 149
    1g  -> 19
    0g  -> 09 atau 9, terutama bila baris juga memuat 0 persen AKG.

    Aturan ini hanya diterapkan pada field berbasis gram. Energi dan natrium tidak disentuh.
    """
    clean = normalize_ocr_text(line)
    if field not in GRAM_BASED_FIELDS:
        return clean

    if " g" in clean or re.search(r"\d+\s*g\b", clean):
        return clean

    field_markers = {
        "lemak_total": ("lemak total",),
        "lemak_jenuh": ("lemak jenuh",),
        "protein": ("protein",),
        "karbohidrat": ("karbohidrat total", "karbohidrat"),
        "gula": ("gula",),
        "garam": ("garam",),
    }

    if not any(marker in clean for marker in field_markers.get(field, ())):
        return clean

    def fix_number_token(match: re.Match) -> str:
        token = match.group(1)
        after = match.group(2) or ""

        # Jangan ubah angka yang memang persen.
        if after.strip().startswith("%"):
            return match.group(0)

        # 2.59 pada label gizi umumnya berasal dari 2.5g.
        if re.fullmatch(r"\d+\.\d+9", token):
            return token[:-1] + " g" + after

        # 09 dan 0 9 berasal dari 0g.
        if token in {"09", "0 9", "0.9"}:
            return "0 g" + after

        # 59 -> 5g, 149 -> 14g, 19 -> 1g.
        if token.isdigit() and len(token) >= 2 and token.endswith("9"):
            return token[:-1] + " g" + after

        # Khusus baris protein 0g yang sering terbaca sebagai 9 dan disertai 0 persen.
        if field == "protein" and token == "9" and re.search(r"\b0\s*%", clean):
            return "0 g" + after

        return match.group(0)

    # Ambil angka setelah marker field agar angka pada teks lain tidak berubah.
    earliest_marker_pos = None
    markers = field_markers.get(field, ())
    for marker in markers:
        idx = clean.find(marker)
        if idx >= 0:
            end = idx + len(marker)
            if earliest_marker_pos is None or end < earliest_marker_pos:
                earliest_marker_pos = end

    if earliest_marker_pos is None:
        return clean

    before = clean[:earliest_marker_pos]
    after = clean[earliest_marker_pos:]
    after = re.sub(r"\b(\d+(?:\.\d+)?|0\s+9)(\s*(?:%|$|\d+\s*%))", fix_number_token, after)
    after = re.sub(r"\s+", " ", after)
    return (before + after).strip()


def _is_suspicious_gram_value(field: str, value: float, line: str) -> bool:
    """Menandai nilai gram yang kemungkinan besar masih berasal dari salah baca satuan g sebagai 9."""
    clean = normalize_ocr_text(line)
    if field not in GRAM_BASED_FIELDS:
        return False
    if re.search(r"\d+\s*g\b", clean):
        return False

    # Nilai gram yang berakhiran .9 setelah repair lama hampir selalu salah baca unit.
    if abs(value - round(value)) > 0:
        frac_digit = int(round((value - int(value)) * 10))
        if frac_digit == 9:
            return True

    # Pada label makanan ringan, protein 9 g tanpa satuan jelas dan persen 0 biasanya berasal dari 0g.
    if field == "protein" and value == 9 and re.search(r"\b0\s*%", clean):
        return True

    return False

def _variant_score(lines: List[str], items: List[Dict[str, Any]], mode: str) -> float:
    raw = " ".join(lines)
    clean = normalize_ocr_text(raw)
    if not clean:
        return 0.0

    nutrition_words = [
        "takaran saji",
        "energi",
        "lemak total",
        "lemak jenuh",
        "protein",
        "karbohidrat",
        "gula",
        "natrium",
        "garam",
    ]
    composition_words = ["komposisi", "pati", "minyak", "pengawet", "gula", "garam", "bumbu"]
    words = composition_words if mode == "composition" else nutrition_words
    keyword_score = sum(1 for word in words if word in clean)
    digit_score = min(len(re.findall(r"\d", clean)) / 20.0, 4.0)
    text_score = min(len(clean) / 250.0, 5.0)
    conf_score = 0.0
    if items:
        conf_score = sum(float(item.get("conf", 0.0)) for item in items) / max(1, len(items))
    return keyword_score * 3.0 + digit_score + text_score + conf_score


def run_ocr_multi_variant(
    reader: Any,
    pil_image: Image.Image,
    min_confidence: float = 0.16,
    max_variants: int = 2,
    mode: str = "nutrition",
) -> Dict[str, Any]:
    """Menjalankan EasyOCR per variasi dan memilih variasi terbaik sebagai raw OCR utama."""
    variants = preprocess_image_variants_for_ocr(pil_image)
    selected_variants = dict(list(variants.items())[:max_variants])

    variant_payloads: List[Dict[str, Any]] = []
    errors: List[str] = []

    for variant_name, image_variant in selected_variants.items():
        results, error_message = _reader_readtext_safely(reader, image_variant)
        if error_message:
            errors.append(f"{variant_name}: {error_message}")
            continue

        variant_items: List[Dict[str, Any]] = []
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
                variant_items.append({
                    "variant": variant_name,
                    "bbox": bbox,
                    "text": clean_text,
                    "conf": confidence,
                })

        lines = group_ocr_results_into_lines(variant_items)
        score = _variant_score(lines, variant_items, mode=mode)
        variant_payloads.append({
            "name": variant_name,
            "items": variant_items,
            "lines": lines,
            "score": score,
        })

    if variant_payloads:
        best_payload = max(variant_payloads, key=lambda payload: payload["score"])
    else:
        best_payload = {"name": "none", "items": [], "lines": [], "score": 0.0}

    all_items: List[Dict[str, Any]] = []
    for payload in variant_payloads:
        all_items.extend(payload.get("items", []))

    return {
        "items": all_items,
        "lines": best_payload.get("lines", []),
        "raw_text": "\n".join(best_payload.get("lines", [])),
        "best_variant": best_payload.get("name", "none"),
        "variant_payloads": variant_payloads,
        "variants": selected_variants,
        "errors": errors,
    }


def _extract_numbers_with_units(line: str) -> List[Tuple[float, str]]:
    clean_line = normalize_ocr_text(line)
    pattern = r"(\d+(?:\.\d+)?)\s*(kkal|kal|mg|g|%)?"
    matches = re.findall(pattern, clean_line)
    values: List[Tuple[float, str]] = []

    for number, unit in matches:
        try:
            values.append((float(number), unit or ""))
        except ValueError:
            continue

    return values


def _numbers_after_marker(line: str, markers: Tuple[str, ...]) -> List[Tuple[float, str]]:
    clean = normalize_ocr_text(line)
    best_pos: Optional[int] = None

    for marker in markers:
        idx = clean.find(marker)
        if idx >= 0:
            pos = idx + len(marker)
            if best_pos is None or pos < best_pos:
                best_pos = pos

    if best_pos is not None:
        clean = clean[best_pos:]

    return _extract_numbers_with_units(clean)


def _choose_number(
    line: str,
    markers: Tuple[str, ...],
    preferred_units: Tuple[str, ...],
    allow_unitless: bool = False,
) -> Optional[float]:
    values = _numbers_after_marker(line, markers)
    if not values:
        return None

    # Utamakan angka dengan satuan yang benar, bukan angka persen AKG.
    for value, unit in values:
        if unit in preferred_units:
            return float(value)

    if allow_unitless:
        for value, unit in values:
            if unit != "%":
                return float(value)

    return None


def _repair_value_by_field(field: str, value: float, line: str) -> Optional[float]:
    """Mencegah angka OCR tidak masuk akal langsung masuk ke form."""
    if value is None:
        return None

    value = float(value)
    clean = normalize_ocr_text(line)

    # Perbaikan spesifik OCR label gizi. Banyak label kecil membuat 2.5 g terbaca 25 g atau 259.
    if field == "lemak_jenuh":
        if value > 100:
            value = value / 100.0
        elif value > 20:
            value = value / 10.0

    elif field == "lemak_total":
        if value > 100:
            value = value / 100.0
        elif value > 50 and "lemak total" in clean:
            value = value / 10.0

    elif field == "karbohidrat":
        if value > 300:
            value = value / 100.0
        elif value > 100:
            value = value / 10.0

    elif field == "gula":
        if value > 200:
            value = value / 100.0
        elif value > 100:
            value = value / 10.0

        # Kasus nyata label: "1 g" sering terbaca menjadi "19" karena huruf g dianggap angka 9.
        # Koreksi ini hanya berlaku bila tidak ada satuan g eksplisit pada baris OCR.
        # Jika tertulis jelas "19 g", nilai tetap dipertahankan karena bisa saja benar.
        has_explicit_gram = bool(re.search(r"\b19\s*g\b", clean))
        has_percent_context = bool(re.search(r"\b\d+\s*%", clean))
        if abs(value - 19.0) < 1e-9 and not has_explicit_gram and has_percent_context:
            value = 1.0

    elif field == "protein":
        if value > 100:
            value = value / 100.0
        elif value > 80:
            value = value / 10.0

    elif field == "garam":
        # Garam label Indonesia sering ditulis sebagai natrium mg. Jangan simpan mg sebagai gram.
        if "mg" in clean and ("natrium" in clean or "sodium" in clean):
            return None
        if value > 50:
            value = value / 1000.0
        elif value > 20:
            value = value / 10.0

    # Lapisan terakhir: bila g masih tersisa sebagai angka 9, koreksi sebelum masuk form.
    if field in GRAM_BASED_FIELDS and _is_suspicious_gram_value(field, value, clean):
        if field == "protein" and value == 9 and re.search(r"\b0\s*%", clean):
            value = 0.0
        elif value >= 10 and abs(value - round(value)) < 1e-9:
            token = str(int(round(value)))
            if token.endswith("9") and len(token) >= 2:
                value = float(token[:-1])
        elif abs(value - round(value, 1)) < 1e-9 and str(round(value, 1)).endswith(".9"):
            value = float(str(round(value, 1))[:-2])

    low, high = NUTRITION_FIELD_LIMITS.get(field, (0.0, float("inf")))
    if value < low or value > high:
        return None

    return round(float(value), 2)


def _candidate_from_line(field: str, line: str) -> Optional[float]:
    clean = normalize_ocr_text(line)
    if field in GRAM_BASED_FIELDS:
        clean = repair_gram_unit_read_as_nine(clean, field)

    if field == "takaran_saji":
        if "takaran saji" not in clean:
            return None
        value = _choose_number(clean, ("takaran saji",), ("g", "ml"), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "energi":
        if "energi" not in clean:
            return None
        if "energi dari lemak" in clean or "from fat" in clean:
            return None
        value = _choose_number(clean, ("energi total", "energi"), ("kkal", "kal"), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "lemak_jenuh":
        if "lemak jenuh" not in clean:
            return None
        value = _choose_number(clean, ("lemak jenuh",), ("g",), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "lemak_total":
        if "lemak total" not in clean:
            return None
        value = _choose_number(clean, ("lemak total",), ("g",), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "protein":
        if "protein" not in clean:
            return None
        value = _choose_number(clean, ("protein",), ("g",), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "karbohidrat":
        if "karbohidrat" not in clean:
            return None
        value = _choose_number(clean, ("karbohidrat total", "karbohidrat"), ("g",), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "gula":
        if "gula" not in clean:
            return None
        value = _choose_number(clean, ("gula",), ("g",), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "natrium_benzoat":
        if "benzoat" not in clean:
            return None
        value = _choose_number(clean, ("natrium benzoat", "benzoat"), ("mg", "g"), allow_unitless=True)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "natrium":
        if "natrium" not in clean and "sodium" not in clean:
            return None
        value = _choose_number(clean, ("natrium", "sodium", "garam"), ("mg",), allow_unitless=False)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    if field == "garam":
        if "garam" not in clean:
            return None
        value = _choose_number(clean, ("garam",), ("g",), allow_unitless=False)
        return _repair_value_by_field(field, value, clean) if value is not None else None

    return None


def _choose_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[float]:
    if not candidates:
        return None

    grouped: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        grouped[round(float(cand["value"]), 2)].append(cand)

    ranked: List[Tuple[int, int, float, float]] = []
    for value, items in grouped.items():
        count = len(items)
        best_variant_rank = min(int(item.get("variant_rank", 99)) for item in items)
        keyword_score = max(float(item.get("keyword_score", 0.0)) for item in items)
        ranked.append((count, -best_variant_rank, keyword_score, value))

    ranked.sort(reverse=True)
    return ranked[0][3]


def parse_nutrition_from_variants(variant_payloads: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Parsing nilai gizi dari beberapa variasi OCR dengan guard logis."""
    data = dict(NUTRITION_DEFAULTS)
    warnings: List[str] = []
    field_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    fields = [
        "takaran_saji",
        "energi",
        "lemak_total",
        "lemak_jenuh",
        "protein",
        "karbohidrat",
        "gula",
        "garam",
        "natrium",
        "natrium_benzoat",
    ]

    ranked_payloads = sorted(variant_payloads, key=lambda payload: payload.get("score", 0), reverse=True)

    for variant_rank, payload in enumerate(ranked_payloads):
        for line in payload.get("lines", []):
            clean = normalize_ocr_text(line)
            for field in fields:
                value = _candidate_from_line(field, clean)
                if value is None:
                    continue
                keyword_score = 1.0
                if field.replace("_", " ") in clean:
                    keyword_score += 1.0
                field_candidates[field].append({
                    "value": value,
                    "line": line,
                    "variant": payload.get("name", "unknown"),
                    "variant_rank": variant_rank,
                    "keyword_score": keyword_score,
                })

    for field in fields:
        chosen = _choose_best_candidate(field_candidates.get(field, []))
        if chosen is not None:
            data[field] = chosen
        elif field != "natrium_benzoat":
            warnings.append(f"{field}: tidak terbaca dengan cukup yakin")

    # Validasi logis: gula adalah bagian dari karbohidrat.
    # Jika gula terbaca lebih besar dari karbohidrat dan angkanya berakhir 9,
    # besar kemungkinan satuan g masih terbaca sebagai angka 9.
    # Contoh: 1g terbaca 19, sementara karbohidrat total 14g.
    try:
        gula_value = float(data.get("gula", 0) or 0)
        karbo_value = float(data.get("karbohidrat", 0) or 0)
        if karbo_value > 0 and gula_value > karbo_value:
            gula_token = str(int(round(gula_value))) if abs(gula_value - round(gula_value)) < 1e-9 else ""
            if gula_token.endswith("9") and len(gula_token) >= 2:
                repaired_gula = float(gula_token[:-1])
                if 0 <= repaired_gula <= karbo_value:
                    data["gula"] = round(repaired_gula, 2)
                    warnings.append("gula: dikoreksi karena satuan g kemungkinan terbaca sebagai angka 9")
    except Exception:
        pass

    # Jika label memakai Garam sebagai gram dan natrium kosong, konversi konservatif ke natrium.
    if data.get("garam", 0) > 0 and data.get("natrium", 0) == 0:
        data["natrium"] = round(float(data["garam"]) * 400.0, 2)

    data["product_name"] = "Produk Tanpa Nama"
    return data, warnings


def parse_nutrition_from_lines(lines: List[str]) -> Dict[str, Any]:
    """Kompatibilitas untuk pemanggilan lama."""
    payload = {"name": "single", "lines": lines, "score": 1.0}
    data, _ = parse_nutrition_from_variants([payload])
    return data


def _line_has_composition_marker(line: str) -> bool:
    clean = normalize_ocr_text(line)
    return any(marker in clean for marker in ["komposisi", "bahan"])


def select_best_composition_lines(variant_payloads: List[Dict[str, Any]]) -> List[str]:
    """Memilih satu variasi terbaik agar komposisi tidak dobel dari hasil multi preprocessing."""
    if not variant_payloads:
        return []

    def score(payload: Dict[str, Any]) -> float:
        raw = " ".join(payload.get("lines", []))
        clean = normalize_ocr_text(raw)
        marker_score = 10 if ("komposisi" in clean or "bahan" in clean) else 0
        additive_score = sum(1 for word in ["pengawet", "gula", "garam", "minyak", "pati", "bumbu", "pewarna"] if word in clean)
        length_score = min(len(clean) / 120.0, 8.0)
        return marker_score + additive_score + length_score

    best = max(variant_payloads, key=score)
    return best.get("lines", [])


def _dedupe_composition_segments(text: str) -> str:
    separators = re.split(r"([,;])", text)
    cleaned_parts: List[str] = []
    seen = set()

    current = ""
    for token in separators:
        if token in [",", ";"]:
            segment = current.strip()
            segment_key = re.sub(r"[^a-z0-9]+", "", normalize_ocr_text(segment))[:80]
            if segment and segment_key and segment_key not in seen:
                cleaned_parts.append(segment)
                seen.add(segment_key)
            current = ""
        else:
            current += token

    tail = current.strip()
    tail_key = re.sub(r"[^a-z0-9]+", "", normalize_ocr_text(tail))[:80]
    if tail and tail_key and tail_key not in seen:
        cleaned_parts.append(tail)

    if not cleaned_parts:
        return text

    return ", ".join(cleaned_parts)



def parse_composition_from_lines(lines: List[str]) -> str:
    """Mengekstrak komposisi dari satu variasi OCR terbaik tanpa menggandakan hasil antar variasi."""
    raw_text = " ".join(lines).strip()
    if not raw_text:
        return "Tidak terdeteksi."

    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    lower_text = raw_text.lower()

    # Cari marker pada teks asli agar tanda koma komposisi tidak berubah menjadi titik.
    marker_match = re.search(r"(?:komposisi|bahan(?:\s*bahan)?|ingredients?)\s*:?", lower_text)
    if marker_match:
        composition = raw_text[marker_match.end():]
    else:
        composition = raw_text

    # Hentikan sebelum bagian Inggris atau metadata kemasan agar komposisi tidak dobel.
    stop_patterns = [
        "ingredienta",
        "ingredients",
        "ingredient",
        "informasi nilai gizi",
        "nutrition information",
        "nutrition facts",
        "nutrition fact",
        "imported",
        "distributed",
        "diproduksi",
        "diedarkan",
        "baik digunakan",
        "expired",
        "exp",
        "tanggal",
        "berat bersih",
        "net weight",
        "netto",
        "barcode",
    ]

    lower_composition = composition.lower()
    cut_index = len(composition)
    for stop_word in stop_patterns:
        idx = lower_composition.find(stop_word)
        if idx >= 0:
            cut_index = min(cut_index, idx)

    composition = composition[:cut_index]
    composition = re.sub(r"\s+", " ", composition).strip(" .,:;")
    composition = _dedupe_composition_segments(composition)
    composition = re.sub(r"\s+", " ", composition).strip(" .,:;")

    if len(composition) < 5:
        return "Tidak terdeteksi."

    return composition[:1].upper() + composition[1:]

def parse_scan_result(reader: Any, pil_image: Image.Image, mode: str = "nutrition") -> Dict[str, Any]:
    """Fungsi praktis untuk dipakai di app.py."""
    ocr_payload = run_ocr_multi_variant(reader, pil_image, mode=mode)
    variant_payloads = ocr_payload.get("variant_payloads", [])

    if mode == "composition":
        selected_lines = select_best_composition_lines(variant_payloads)
        parsed = dict(NUTRITION_DEFAULTS)
        parsed["komposisi"] = parse_composition_from_lines(selected_lines)
        quality_warnings: List[str] = []
        if parsed["komposisi"] == "Tidak terdeteksi.":
            quality_warnings.append("Komposisi belum terbaca jelas. Coba crop hanya area komposisi atau input manual.")
        raw_text = "\n".join(selected_lines)
    else:
        parsed, quality_warnings = parse_nutrition_from_variants(variant_payloads)
        raw_text = ocr_payload.get("raw_text", "")

    return {
        "parsed": parsed,
        "lines": raw_text.splitlines() if raw_text else [],
        "raw_text": raw_text,
        "items": ocr_payload.get("items", []),
        "variants": ocr_payload.get("variants", {}),
        "best_variant": ocr_payload.get("best_variant", "none"),
        "variant_payloads": variant_payloads,
        "errors": ocr_payload.get("errors", []),
        "quality_warnings": quality_warnings,
    }
