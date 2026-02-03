#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
예시용: 초본 vs 수정본(텍스트) 비교 리포트
- 표현 유사도 (surface)
- 의미 유사도 (semantic: TF-IDF cosine, 문단 단위 분포)
- 구조 변화도 (structure: 제목/구획 추정 후 트리 수준 비교)
- 정보 확장·축소 (info: 숫자/인용/괄호/고유토큰 등 단순 프록시)

입력: 두 텍스트(파일 또는 문자열)
출력: JSON 리포트 + 콘솔 요약
"""

import re
import json
import math
import argparse
import difflib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

# 의미 유사도(semantic)용: 표준적인 경량 방식(TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Utilities
# -----------------------------
def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # 과도한 공백 정리(의도적으로 강하게 정리하면 diff 왜곡됨 -> 약하게)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_paragraphs(s: str) -> List[str]:
    s = normalize_text(s)
    if not s:
        return []
    parts = re.split(r"\n\s*\n", s)
    return [p.strip() for p in parts if p.strip()]


def tokenize_simple(s: str) -> List[str]:
    # 공백 기준 + 기호 제거(아주 단순)
    s = normalize_text(s)
    s = re.sub(r"[^\w가-힣]+", " ", s, flags=re.UNICODE)
    toks = [t for t in s.split() if t]
    return toks


def count_regex(pattern: str, s: str) -> int:
    return len(re.findall(pattern, s))


def safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def safe_quantiles(xs: List[float], qs=(0.1, 0.5, 0.9)) -> Dict[str, float]:
    if not xs:
        return {f"q{int(q*100)}": 0.0 for q in qs}
    ys = sorted(xs)
    out = {}
    for q in qs:
        idx = int(round((len(ys) - 1) * q))
        out[f"q{int(q*100)}"] = ys[idx]
    return out


# -----------------------------
# 1) Surface similarity
# -----------------------------
def surface_metrics(a: str, b: str) -> Dict[str, Any]:
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    # 글자 단위 유사도(빠른 프록시): SequenceMatcher ratio
    char_ratio = difflib.SequenceMatcher(None, a_n, b_n).ratio()

    # 단어 단위 삽입/삭제/치환 비율(대략)
    a_toks = tokenize_simple(a_n)
    b_toks = tokenize_simple(b_n)
    sm = difflib.SequenceMatcher(None, a_toks, b_toks)
    ops = sm.get_opcodes()
    ins = sum((j2 - j1) for tag, i1, i2, j1, j2 in ops if tag == "insert")
    dele = sum((i2 - i1) for tag, i1, i2, j1, j2 in ops if tag == "delete")
    repl = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag == "replace")

    denom = max(1, len(a_toks))
    return {
        "char_similarity": round(char_ratio, 4),
        "token_counts": {"draft": len(a_toks), "final": len(b_toks)},
        "token_change_rates_vs_draft": {
            "insert_rate": round(ins / denom, 4),
            "delete_rate": round(dele / denom, 4),
            "replace_rate": round(repl / denom, 4),
        },
        "length": {
            "chars_draft": len(a_n),
            "chars_final": len(b_n),
            "delta_chars": len(b_n) - len(a_n),
        },
    }


# -----------------------------
# 2) Semantic similarity
# -----------------------------
def paragraph_alignment_semantic(paras_a: List[str], paras_b: List[str]) -> Tuple[List[Tuple[int, int, float]], List[float]]:
    """
    아주 단순한 문단 매칭:
    - draft 문단 하나당 final 문단 중 의미 유사도 최대값을 매칭(중복 허용)
    - 정책 문서용 예시로 충분한 수준
    """
    if not paras_a or not paras_b:
        return [], []

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    # fit은 합쳐서
    X = vec.fit_transform(paras_a + paras_b)
    Xa = X[: len(paras_a)]
    Xb = X[len(paras_a) :]

    sim = cosine_similarity(Xa, Xb)  # (A, B)
    pairs = []
    best_sims = []
    for i in range(sim.shape[0]):
        j = int(sim[i].argmax())
        s = float(sim[i, j])
        pairs.append((i, j, s))
        best_sims.append(s)
    return pairs, best_sims


def semantic_metrics(a: str, b: str) -> Dict[str, Any]:
    paras_a = split_paragraphs(a)
    paras_b = split_paragraphs(b)

    pairs, best_sims = paragraph_alignment_semantic(paras_a, paras_b)

    # 전체 문서 유사도도 추가(텍스트 단위 TF-IDF)
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform([normalize_text(a), normalize_text(b)])
    doc_sim = float(cosine_similarity(X[0], X[1])[0, 0])

    # 분포 기록(단일 점수 “판정” 느낌 줄이기 위해)
    dist = {
        "mean": round(safe_mean(best_sims), 4),
        **{k: round(v, 4) for k, v in safe_quantiles(best_sims).items()},
        "bins": {
            ">=0.8": sum(1 for x in best_sims if x >= 0.8),
            "0.5~0.8": sum(1 for x in best_sims if 0.5 <= x < 0.8),
            "0.2~0.5": sum(1 for x in best_sims if 0.2 <= x < 0.5),
            "<0.2": sum(1 for x in best_sims if x < 0.2),
        },
    }

    # 매칭 로그(상위 일부만)
    top_pairs = sorted(pairs, key=lambda t: t[2], reverse=True)[:10]
    top_pairs_out = [{"draft_para": i, "final_para": j, "sim": round(s, 4)} for i, j, s in top_pairs]

    return {
        "doc_tfidf_cosine": round(doc_sim, 4),
        "para_counts": {"draft": len(paras_a), "final": len(paras_b)},
        "para_bestmatch_similarity_distribution": dist,
        "top_para_matches": top_pairs_out,
    }


# -----------------------------
# 3) Structural change
# -----------------------------
HEADING_PATTERNS = [
    r"^\s*#{1,6}\s+.+$",             # Markdown heading
    r"^\s*\d+(\.\d+)*\s+.+$",        # 1. 1.1 2.3.4 ...
    r"^\s*[IVXLC]+\.\s+.+$",         # Roman numerals
]

def extract_structure_blocks(s: str) -> List[Tuple[str, List[str]]]:
    """
    매우 단순한 구조 추정:
    - 제목(heading) 라인을 구획의 시작으로 보고
    - 각 구획의 본문 라인들을 묶음
    """
    lines = normalize_text(s).split("\n")
    blocks: List[Tuple[str, List[str]]] = []
    cur_title = "__ROOT__"
    cur_body: List[str] = []

    def is_heading(line: str) -> bool:
        for p in HEADING_PATTERNS:
            if re.match(p, line):
                return True
        return False

    for line in lines:
        if is_heading(line):
            # flush
            if cur_title != "__ROOT__" or cur_body:
                blocks.append((cur_title, cur_body))
            cur_title = line.strip()
            cur_body = []
        else:
            cur_body.append(line)

    # last
    if cur_title != "__ROOT__" or cur_body:
        blocks.append((cur_title, cur_body))

    return blocks


def structure_metrics(a: str, b: str) -> Dict[str, Any]:
    A = extract_structure_blocks(a)
    B = extract_structure_blocks(b)

    titles_a = [t for t, _ in A if t != "__ROOT__"]
    titles_b = [t for t, _ in B if t != "__ROOT__"]

    set_a = set(titles_a)
    set_b = set(titles_b)

    added = sorted(list(set_b - set_a))[:20]
    removed = sorted(list(set_a - set_b))[:20]
    common = sorted(list(set_a & set_b))

    # 제목 순서 변화(공통 제목들에 대해서만 LCS 기반 근사)
    common_a = [t for t in titles_a if t in set_b]
    common_b = [t for t in titles_b if t in set_a]
    lcs_len = difflib.SequenceMatcher(None, common_a, common_b).find_longest_match(0, len(common_a), 0, len(common_b)).size
    # 주의: 위는 "연속" 최장 일치. 순서 안정성 프록시로 사용(정확 LCS는 아니지만 예시엔 충분)
    order_stability = (lcs_len / max(1, len(common_a))) if common_a else 0.0

    return {
        "heading_counts": {"draft": len(titles_a), "final": len(titles_b)},
        "heading_set_changes": {
            "added_sample": added,
            "removed_sample": removed,
            "common_count": len(common),
        },
        "order_stability_proxy": round(order_stability, 4),
        "note": "구조는 제목 패턴 기반의 단순 추정이며, 형식이 정돈될수록 정확도가 상승합니다.",
    }


# -----------------------------
# 4) Information expansion/reduction
# -----------------------------
def info_metrics(a: str, b: str) -> Dict[str, Any]:
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    def info_features(s: str) -> Dict[str, int]:
        toks = tokenize_simple(s)
        uniq = len(set(toks))
        return {
            "tokens": len(toks),
            "unique_tokens": uniq,
            "numbers": count_regex(r"\d+", s),
            "quotes": count_regex(r"[\"“”‘’']", s),
            "paren_groups": count_regex(r"\([^)]*\)", s),
            "bracket_groups": count_regex(r"\[[^\]]*\]", s),
            "urls": count_regex(r"https?://\S+", s),
            "bullets": count_regex(r"^\s*[-*•]\s+", s, ) if False else 0,  # (필요하면 확장)
        }

    fa = info_features(a_n)
    fb = info_features(b_n)

    delta = {k: fb[k] - fa[k] for k in fa.keys()}
    # 비율도 제공(0 division 방지)
    rate = {k: round((fb[k] / max(1, fa[k])), 4) for k in fa.keys()}

    return {
        "draft": fa,
        "final": fb,
        "delta": delta,
        "ratio_final_over_draft": rate,
        "note": "정보 확장/축소는 사실성·품질 판단이 아닌, 정보 요소의 증감 기록(프록시)입니다.",
    }


# -----------------------------
# Report wrapper
# -----------------------------
def build_report(draft: str, final: str) -> Dict[str, Any]:
    draft_n = normalize_text(draft)
    final_n = normalize_text(final)

    report = {
        "inputs": {
            "draft_chars": len(draft_n),
            "final_chars": len(final_n),
        },
        "metrics": {
            "expression_similarity": surface_metrics(draft_n, final_n),
            "semantic_similarity": semantic_metrics(draft_n, final_n),
            "structural_change": structure_metrics(draft_n, final_n),
            "information_expansion_reduction": info_metrics(draft_n, final_n),
        },
        "policy_notes": [
            "본 리포트는 AI 사용 여부 판별, 기여도 평가, 품질 평가를 수행하지 않습니다.",
            "모든 지표는 초본 대비 최종본의 변형을 설명하는 기록 데이터입니다.",
            "총점·등급·우열 판단을 산출하지 않습니다.",
        ],
    }
    return report


def read_text_arg(path_or_literal: str) -> str:
    # 파일이 존재하면 파일로 읽고, 아니면 문자열로 간주
    try:
        with open(path_or_literal, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return path_or_literal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="초본 텍스트 또는 파일 경로")
    ap.add_argument("--final", required=True, help="수정본 텍스트 또는 파일 경로")
    ap.add_argument("--out", default="", help="JSON 저장 경로(옵션)")
    args = ap.parse_args()

    draft = read_text_arg(args.draft)
    final = read_text_arg(args.final)

    rep = build_report(draft, final)

    # 콘솔 요약(평가처럼 보이지 않게, 항목만 출력)
    expr = rep["metrics"]["expression_similarity"]["char_similarity"]
    sem = rep["metrics"]["semantic_similarity"]["doc_tfidf_cosine"]
    struct_cnt = rep["metrics"]["structural_change"]["heading_counts"]
    info_delta = rep["metrics"]["information_expansion_reduction"]["delta"]

    print("=== 변형 리포트(요약) ===")
    print(f"- 표현 유사도(문자): {expr}")
    print(f"- 의미 유사도(TF-IDF 코사인): {sem}")
    print(f"- 구조(제목 수): draft={struct_cnt['draft']}, final={struct_cnt['final']}")
    print(f"- 정보 변화(토큰/고유토큰/숫자): "
          f"{info_delta['tokens']:+d}, {info_delta['unique_tokens']:+d}, {info_delta['numbers']:+d}")
    print()

    js = json.dumps(rep, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"[saved] {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()
