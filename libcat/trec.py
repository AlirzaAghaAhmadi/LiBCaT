from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .text_utils import clean_text, join_nonempty


def load_topics(path: str | Path, template_contains: str | None = "breast") -> pd.DataFrame:
    root = ET.parse(path).getroot()
    rows = []
    for topic in root.findall(".//topic"):
        template = topic.attrib.get("template", "")
        if template_contains and template_contains.lower() not in template.lower():
            continue
        fields = []
        field_map = {}
        for field in topic.findall("field"):
            name = clean_text(field.attrib.get("name"))
            value = clean_text(field.text)
            field_map[name] = value
            if value:
                fields.append(f"{name}: {value}")
        rows.append({
            "topic_id": str(topic.attrib.get("number", "")),
            "template": template,
            "query": ". ".join(fields),
            "fields": field_map,
        })
    return pd.DataFrame(rows)


def load_qrels(path: str | Path) -> pd.DataFrame:
    # Standard qrels: topic iteration docid grade
    q = pd.read_csv(path, sep=r"\s+", names=["topic_id", "iteration", "nct_id", "grade"], dtype={"topic_id": str, "nct_id": str})
    q["grade"] = q["grade"].astype(int)
    return q


def dcg(grades: list[int]) -> float:
    """TREC Clinical Trials 2023 DCG with *linear* gains (0, 1, 2).

    The official track gives gain 1 to excluded and 2 to eligible trials.
    An exponential 2**grade - 1 transformation would incorrectly weight
    eligible documents as 3 and would not reproduce the reported metric.
    """
    return sum(float(g) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked_ids: list[str], grade_map: dict[str, int], k: int = 10) -> float:
    grades = [int(grade_map.get(doc, 0)) for doc in ranked_ids[:k]]
    ideal = sorted(grade_map.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(grades) / denom if denom else 0.0


def eligible_hit_at_k(ranked_ids: list[str], grade_map: dict[str, int], k: int = 5) -> float:
    return float(any(grade_map.get(doc, 0) >= 2 for doc in ranked_ids[:k]))


def eligible_precision_at_k(ranked_ids: list[str], grade_map: dict[str, int], k: int = 10) -> float:
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(1 for doc in top if grade_map.get(doc, 0) >= 2) / k


def eligible_recall_at_k(ranked_ids: list[str], grade_map: dict[str, int], k: int = 10) -> float:
    eligible = {doc for doc, grade in grade_map.items() if grade >= 2}
    if not eligible:
        return 0.0
    return len(eligible.intersection(ranked_ids[:k])) / len(eligible)


def eligible_mrr(ranked_ids: list[str], grade_map: dict[str, int]) -> float:
    for i, doc in enumerate(ranked_ids, start=1):
        if grade_map.get(doc, 0) >= 2:
            return 1.0 / i
    return 0.0


def qrel_coverage(candidate_ids: set[str], grade_map: dict[str, int], min_grade: int = 2) -> float:
    judged = {doc for doc, grade in grade_map.items() if grade >= min_grade}
    return len(judged & candidate_ids) / len(judged) if judged else 0.0
