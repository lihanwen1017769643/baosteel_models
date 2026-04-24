from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional


DATE_RANGE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2}).*?(20\d{2}-\d{2}-\d{2})")
OUTLET_RE = re.compile(r"(DA\d{3}[_#A-Za-z0-9\-]*)")


def extract_metadata_from_path(file_path: Path) -> Dict[str, Optional[str]]:
    folder_name = file_path.parent.parent.name if file_path.parent.name.lower() == "null" else file_path.parent.name
    file_name = file_path.name

    match = DATE_RANGE_RE.search(file_name)
    start_date, end_date = (match.group(1), match.group(2)) if match else (None, None)

    outlet_match = OUTLET_RE.search(file_name)
    outlet_id = outlet_match.group(1) if outlet_match else None

    process_hint = None
    if "机头" in file_name or "烧结" in file_name:
        process_hint = "烧结机头"
    elif "二次" in file_name:
        process_hint = "转炉二次"
    elif "矿槽" in file_name:
        process_hint = "高炉矿槽"
    elif "出铁" in file_name:
        process_hint = "高炉出铁"

    return {
        "source_folder": folder_name,
        "source_file": file_name,
        "point_name_hint": folder_name,
        "outlet_id": outlet_id,
        "process_hint": process_hint,
        "file_start_date": start_date,
        "file_end_date": end_date,
    }


def parse_title_metadata(title_text: str) -> Dict[str, Optional[str]]:
    if not title_text:
        return {"title_outlet": None, "title_start_time": None, "title_end_time": None}

    outlet_match = OUTLET_RE.search(title_text)
    dt_match = re.search(
        r"(20\d{2})年(\d{2})月(\d{2})日(\d{2})时(\d{2})分至(20\d{2})年(\d{2})月(\d{2})日(\d{2})时(\d{2})分",
        title_text,
    )

    def _fmt(groups, idx):
        return f"{groups[idx]}-{groups[idx+1]}-{groups[idx+2]} {groups[idx+3]}:{groups[idx+4]}:00"

    start_time, end_time = None, None
    if dt_match:
        g = dt_match.groups()
        start_time = _fmt(g, 0)
        end_time = _fmt(g, 5)

    return {
        "title_outlet": outlet_match.group(1) if outlet_match else None,
        "title_start_time": start_time,
        "title_end_time": end_time,
    }
