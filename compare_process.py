#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
예시용: 초본 vs 수정본(텍스트) 비교 리포트
- 표현 유사도 (surface)
- 의미 유사도 (semantic: TF-IDF cosine, 구간 단위 분포)
- 구조 변화도 (structure: 제목/구획 추정 후 비교)
- 정보 확장·축소 (info: 숫자/인용/괄호/고유토큰 등 단순 프록시)

입력: 두 텍스트(파일 또는 문자열)
출력: JSON 리포트 + 콘솔 요약
"""

import re
import json
import argparse
import difflib
from typing import List, Tuple, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


HEADING_PATTERNS = [
    r"^\s*#{1,6}\s+.+$",             # Markdown heading
    r"^\s*\d+(\.\d+)*\s+.+$",        # 1. 1.1 2.3.4 ...
    r"^\s*[IVXLC]+\.\s+.+$",         # Roman numerals
]


def is_heading_line(line: str) -> bool:
    for p in HEADING_PATTERNS:
        if re.match(p, line):
            return True
    return False


def split_blocks(s: str) -> List[str]:
    """
    비교 단위(블록) 분할:
    - 기본은 빈 줄 기준
    - 제목 라인도 블록 경계로 취급(평문인데 번호/제목 섞인 문서 안정화)
    """
    s = normalize_text(s)
    if not s:
        return []

    lines = s.split("\n")
    chunks: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        txt = "\n".join(buf).strip()
        if txt:
            chunks.append(txt)
        buf = []

    for line in lines:
        if is_heading_line(line):
            flush()
            buf.append(line)
            continue

        if re.match(r"^\s*$", line):
            flush()
            continue

        buf.append(line)

    flush()
    return chunks


def tokenize_simple(s: str) -> List[str]:
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


def surface_metrics(a: str, b: str) -> Dict[str, Any]:
    a_n = normalize_text(a)
    b_n = normalize_text(b)

    char_ratio = difflib.SequenceMatcher(None, a_n, b_n).ratio()

    a_toks = tokenize_simple(a_n)
    b_toks = tokenize_simple(b_n)
    sm = difflib.SequenceMatcher(None, a_toks, b_toks)
    ops = sm.get_opcodes()

    ins = sum((j2 - j1) for tag, i1, i2, j1, j2 in ops if tag == "insert")
    dele = sum((i2 - i1) for tag, i1, i2, j1, j2 in ops if tag == "delete")

    # replace는 “대체 구간 길이”를 초안/최종 각각 기록(해석 왜곡 최소화)
    repl_a = sum((i2 - i1) for tag, i1, i2, j1, j2 in ops if tag == "replace")
    repl_b = sum((j2 - j1) for tag, i1, i2, j1, j2 in ops if tag == "replace")

    denom = max(1, len(a_toks))
    return {
        "char_similarity": round(char_ratio, 4),
        "token_counts": {"draft": len(a_toks), "final": len(b_toks)},
        "token_change_rates_vs_draft": {
            "insert_rate": round(ins / denom, 4),
            "delete_rate": round(dele / denom, 4),
            "replace_rate_draft_side": round(repl_a / denom, 4),
            "replace_rate_final_side": round(repl_b / denom, 4),
        },
        "length": {
            "chars_draft": len(a_n),
            "chars_final": len(b_n),
            "delta_chars": len(b_n) - len(a_n),
        },
    }


def paragraph_alignment_semantic(blocks_a: List[str], blocks_b: List[str]) -> Tuple[List[Tuple[int, int, float]], List[float]]:
    """
    단순 블록 매칭:
    - draft 블록 i마다 final 블록 j 중 최고 유사도를 매칭(중복 허용)
    - 분할/병합 정렬까지는 안 하지만, “경계(빈줄/제목)” 반영으로 안정성 향상
    """
    if not blocks_a or not blocks_b:
        return [], []

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform(blocks_a + blocks_b)
    Xa = X[: len(blocks_a)]
    Xb = X[len(blocks_a) :]

    sim = cosine_similarity(Xa, Xb)
    pairs = []
    best_sims = []
    for i in range(sim.shape[0]):
        j = int(sim[i].argmax())
        s = float(sim[i, j])
        pairs.append((i, j, s))
        best_sims.append(s)
    return pairs, best_sims


def semantic_metrics(a: str, b: str) -> Dict[str, Any]:
    blocks_a = split_blocks(a)
    blocks_b = split_blocks(b)

    pairs, best_sims = paragraph_alignment_semantic(blocks_a, blocks_b)

    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform([normalize_text(a), normalize_text(b)])
    doc_sim = float(cosine_similarity(X[0], X[1])[0, 0])

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

    top_pairs = sorted(pairs, key=lambda t: t[2], reverse=True)[:10]
    top_pairs_out = [{"draft_block": i, "final_block": j, "sim": round(s, 4)} for i, j, s in top_pairs]

    return {
        "doc_tfidf_cosine": round(doc_sim, 4),
        "block_counts": {"draft": len(blocks_a), "final": len(blocks_b)},
        "block_bestmatch_similarity_distribution": dist,
        "top_block_matches": top_pairs_out,
        "note": "의미 유사도는 TF-IDF 기반 키워드 유사도이며, 블록(빈줄/제목 경계) 단위로 최대 매칭 분포를 기록합니다.",
    }


def extract_structure_titles(s: str) -> List[str]:
    lines = normalize_text(s).split("\n")
    titles = [line.strip() for line in lines if is_heading_line(line)]
    return titles


def lcs_length(a: List[str], b: List[str]) -> int:
    # 제목 수는 보통 작아서 O(nm) DP로 충분
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            temp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def structure_metrics(a: str, b: str) -> Dict[str, Any]:
    titles_a = extract_structure_titles(a)
    titles_b = extract_structure_titles(b)

    set_a = set(titles_a)
    set_b = set(titles_b)

    added = sorted(list(set_b - set_a))[:20]
    removed = sorted(list(set_a - set_b))[:20]
    common = sorted(list(set_a & set_b))

    common_a = [t for t in titles_a if t in set_b]
    common_b = [t for t in titles_b if t in set_a]
    lcs_len = lcs_length(common_a, common_b)
    order_stability = (lcs_len / max(1, len(common_a))) if common_a else 0.0

    return {
        "heading_counts": {"draft": len(titles_a), "final": len(titles_b)},
        "heading_set_changes": {
            "added_sample": added,
            "removed_sample": removed,
            "common_count": len(common),
        },
        "order_stability_proxy": round(order_stability, 4),
        "note": "구조는 제목 패턴 기반 추정이며, 순서 안정성은 공통 제목에 대한 LCS 기반 프록시입니다.",
    }


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
            "bullets": count_regex(r"(?m)^\s*[-*•]\s+", s),
        }

    fa = info_features(a_n)
    fb = info_features(b_n)

    delta = {k: fb[k] - fa[k] for k in fa.keys()}
    rate = {k: round((fb[k] / max(1, fa[k])), 4) for k in fa.keys()}

    return {
        "draft": fa,
        "final": fb,
        "delta": delta,
        "ratio_final_over_draft": rate,
        "note": "정보 확장/축소는 사실성·품질 판단이 아닌, 정보 요소의 증감 기록(프록시)입니다.",
    }


def build_report(draft: str, final: str) -> Dict[str, Any]:
    draft_n = normalize_text(draft)
    final_n = normalize_text(final)

    return {
        "inputs": {"draft_chars": len(draft_n), "final_chars": len(final_n)},
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


def read_text_arg(path_or_literal: str) -> str:
    # 파일로 읽기 시도 → 실패하면 문자열로 간주
    try:
        with open(path_or_literal, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, IsADirectoryError):
        return path_or_literal
    except UnicodeDecodeError:
        # 인코딩 문제면 사용자에게 명확히 알리는 게 안전
        raise RuntimeError(f"파일 인코딩(utf-8)으로 읽기 실패: {path_or_literal}")
    except PermissionError:
        raise RuntimeError(f"파일 권한 문제로 읽기 실패: {path_or_literal}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="초본 텍스트 또는 파일 경로")
    ap.add_argument("--final", required=True, help="수정본 텍스트 또는 파일 경로")
    ap.add_argument("--out", default="", help="JSON 저장 경로(옵션)")
    args = ap.parse_args()

    draft = read_text_arg(args.draft)
    final = read_text_arg(args.final)

    rep = build_report(draft, final)

    expr = rep["metrics"]["expression_similarity"]["char_similarity"]
    sem = rep["metrics"]["semantic_similarity"]["doc_tfidf_cosine"]
    struct_cnt = rep["metrics"]["structural_change"]["heading_counts"]
    info_delta = rep["metrics"]["information_expansion_reduction"]["delta"]

    print("=== 변형 리포트(요약) ===")
    print(f"- 표현 유사도(문자): {expr}")
    print(f"- 의미 유사도(TF-IDF 코사인): {sem}")
    print(f"- 구조(제목 수): draft={struct_cnt['draft']}, final={struct_cnt['final']}")
    print(
        f"- 정보 변화(토큰/고유토큰/숫자): "
        f"{info_delta['tokens']:+d}, {info_delta['unique_tokens']:+d}, {info_delta['numbers']:+d}"
    )
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
