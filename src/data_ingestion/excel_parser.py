from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data_ingestion.metadata import extract_metadata_from_path, parse_title_metadata
from src.utils.io_utils import ensure_dir


POLLUTANTS = ["烟尘", "二氧化硫", "氮氧化物"]


def _normalize_token(x: str) -> str:
    x = str(x).replace("\n", "").replace(" ", "")
    x = x.replace("（", "(").replace("）", ")")
    x = x.replace("m³", "m3")
    return x


def _is_empty_token(x: str) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower().startswith("unnamed") or s.lower() == "nan"


def _flatten_header_tokens(tokens: List[str]) -> str:
    ts = [_normalize_token(t) for t in tokens if not _is_empty_token(t)]
    if not ts:
        return ""

    joined = "_".join(dict.fromkeys(ts))

    if "监控时间" in joined:
        return "监控时间"
    if "生产设施工况" in joined:
        return "生产设施工况"

    if "废气" in joined and "排放量" in joined:
        return "废气排放量m3"

    for p in POLLUTANTS:
        if p in joined:
            if "设备维护标记" in joined:
                if "自动" in joined:
                    return f"{p}_设备维护标记_自动"
                if "人工" in joined:
                    return f"{p}_设备维护标记_人工"
                return f"{p}_设备维护标记"
            if "排放量" in joined and "kg" in joined:
                return f"{p}_排放量kg"
            if "折算值" in joined:
                return f"{p}_折算值"
            if "实测值" in joined or "浓度" in joined:
                return f"{p}_实测值"

    for factor in ["烟气流速", "烟气温度", "烟气湿度", "烟气压力", "氧含量"]:
        if factor in joined:
            return factor

    cleaned = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fa5]", "", joined)
    return cleaned[:80]


def _detect_header_and_data_start(raw_df: pd.DataFrame) -> Tuple[int, int]:
    header_start = None
    for i in range(min(len(raw_df), 30)):
        row = raw_df.iloc[i].astype(str)
        if row.str.contains("监控时间", na=False).any():
            header_start = i
            break
    if header_start is None:
        raise ValueError("未识别到表头中的监控时间字段")

    data_start = None
    first_col = raw_df.iloc[:, 0]
    for i in range(header_start + 1, min(len(raw_df), header_start + 20)):
        val = pd.to_datetime(first_col.iloc[i], errors="coerce")
        if pd.notna(val):
            data_start = i
            break
    if data_start is None:
        data_start = header_start + 3

    return header_start, data_start


def parse_single_excel(file_path: Path) -> pd.DataFrame:
    try:
        raw_df = pd.read_excel(file_path, header=None)
    except ValueError as e:
        if "format cannot be determined" in str(e):
            raw_df = pd.read_excel(file_path, header=None, engine="xlrd")
        else:
            raise
    title_text = str(raw_df.iloc[0, 0]) if len(raw_df) else ""

    header_start, data_start = _detect_header_and_data_start(raw_df)
    header_df = raw_df.iloc[header_start:data_start]

    col_names = []
    for c in range(raw_df.shape[1]):
        tokens = [header_df.iloc[r, c] for r in range(header_df.shape[0])]
        name = _flatten_header_tokens(tokens)
        if not name:
            name = f"空列_{c}"
        col_names.append(name)

    data = raw_df.iloc[data_start:].copy()
    data.columns = col_names

    data = data.dropna(axis=1, how="all")
    data = data.loc[:, ~data.columns.duplicated()]
    data = data[[c for c in data.columns if not c.startswith("空列_")]]

    meta = extract_metadata_from_path(file_path)
    meta.update(parse_title_metadata(title_text))

    for k, v in meta.items():
        data[k] = v
    data["source_path"] = str(file_path)

    return data.reset_index(drop=True)


def _cache_name_for_file(file_path: Path) -> str:
    digest = hashlib.md5(str(file_path).encode("utf-8")).hexdigest()  # noqa: S324
    return f"{digest}.pkl"


def parse_all_excels(
    data_root: Path,
    file_glob: str = "**/*.xls",
    max_files: int | None = None,
    batch_size: int = 20,
    cache_dir: Path | None = None,
    resume_parse: bool = False,
) -> pd.DataFrame:
    files = sorted([p for p in data_root.glob(file_glob) if p.is_file()])
    if max_files is not None:
        if len(files) > max_files:
            # Round-robin select across parent directories for better point/process coverage.
            bucket: Dict[str, List[Path]] = {}
            for fp in files:
                key = str(fp.parent)
                bucket.setdefault(key, []).append(fp)
            selected: List[Path] = []
            while len(selected) < max_files:
                advanced = False
                for key in sorted(bucket.keys()):
                    if bucket[key]:
                        selected.append(bucket[key].pop(0))
                        advanced = True
                        if len(selected) >= max_files:
                            break
                if not advanced:
                    break
            files = sorted(selected)

    cache_root = None
    if cache_dir is not None:
        cache_root = ensure_dir(cache_dir)

    frames = []
    errors: Dict[str, str] = {}
    for i, p in enumerate(files, 1):
        try:
            loaded_from_cache = False
            cache_file = None
            if cache_root is not None:
                cache_file = cache_root / _cache_name_for_file(p)
                if resume_parse and cache_file.exists():
                    frames.append(pd.read_pickle(cache_file))
                    loaded_from_cache = True

            if not loaded_from_cache:
                parsed = parse_single_excel(p)
                frames.append(parsed)
                if cache_file is not None:
                    parsed.to_pickle(cache_file)
        except Exception as e:  # noqa: BLE001
            errors[str(p)] = str(e)
        if i % max(batch_size, 1) == 0:
            print(f"[parse] processed {i}/{len(files)} files")

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if errors:
        err_df = pd.DataFrame({"file": list(errors.keys()), "error": list(errors.values())})
        err_df.to_csv("outputs/tables/parse_errors.csv", index=False, encoding="utf-8-sig")
    else:
        err_path = Path("outputs/tables/parse_errors.csv")
        if err_path.exists():
            err_path.unlink()
    return all_df


def load_process_mapping(mapping_file: Path) -> pd.DataFrame:
    mdf = pd.read_excel(mapping_file)
    mdf = mdf.rename(columns={"排口编号及排口名称": "mapping_outlet_name", "治理工艺": "治理工艺"})
    mdf["outlet_id"] = mdf["mapping_outlet_name"].astype(str).str.extract(r"(DA\d{3}[_#A-Za-z0-9\-]*)")
    return mdf
