"""
DepMap 26Q1 Feature Extractor
Usage:
  python extract.py --features cnv expr --genes TP53 KRAS --output subset.csv
  python extract.py --features mut_lof --lineage Lung Breast --output lof_lung_breast.csv
  python extract.py --features meta --list-lineages
"""

import argparse
import sys
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "DepMap_26Q1_preprocessed")

FILE_MAP = {
    "cnv":        ("cnv_log2.csv",           "CNV_"),
    "expr":       ("expression_zscore.csv",  "EXP_"),
    "expr_tpm":   ("expression_logTPM.csv",  "EXP_"),
    "mut":        ("mutation_binary.csv",    "MUT_"),
    "mut_lof":    ("mutation_lof.csv",       "MUT_LoF_"),
    "mut_hot":    ("mutation_hotspot.csv",   "MUT_HOT_"),
    "meta":       ("metadata_encoded.csv",   None),
}

META_RAW = "metadata_raw.csv"


def load_meta_raw(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, META_RAW)
    return pd.read_csv(path, index_col="ModelID")


def filter_samples(meta_raw: pd.DataFrame, lineage, disease, model_ids) -> list:
    mask = pd.Series(True, index=meta_raw.index)
    if lineage:
        mask &= meta_raw["OncotreeLineage"].isin(lineage)
    if disease:
        mask &= meta_raw["OncotreePrimaryDisease"].isin(disease)
    if model_ids:
        mask &= meta_raw.index.isin(model_ids)
    selected = meta_raw.index[mask].tolist()
    if not selected:
        sys.exit("오류: 지정한 필터 조건에 맞는 세포주가 없습니다.")
    return selected


def load_feature_block(data_dir: str, feature: str, genes: list, samples: list) -> pd.DataFrame:
    filename, prefix = FILE_MAP[feature]
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        sys.exit(f"오류: 파일 없음 — {path}")

    # 필요한 유전자 컬럼만 읽기 (메모리 절약)
    if genes and prefix:
        target_cols = [prefix + g for g in genes]
        try:
            df = pd.read_csv(path, index_col=0, usecols=lambda c: c == "ModelID" or c in target_cols)
        except Exception:
            df = pd.read_csv(path, index_col=0)
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            print(f"  [경고] {feature}: {len(missing)}개 유전자를 찾지 못함 — {missing[:5]}{'...' if len(missing)>5 else ''}")
        df = df[[c for c in target_cols if c in df.columns]]
    elif feature == "meta":
        df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)

    # 샘플 필터
    available = [s for s in samples if s in df.index]
    if len(available) < len(samples):
        print(f"  [경고] {feature}: {len(samples)-len(available)}개 샘플이 파일에 없음")
    return df.loc[available]


def list_lineages(data_dir: str):
    meta = load_meta_raw(data_dir)
    counts = meta["OncotreeLineage"].value_counts()
    print(f"\n{'암종 계통 (OncotreeLineage)':<45} {'세포주 수':>8}")
    print("-" * 55)
    for lineage, n in counts.items():
        print(f"  {lineage:<43} {n:>8,}")
    print(f"\n  총 {len(meta):,}개 세포주 / {counts.index.nunique()}개 lineage")


def list_diseases(data_dir: str):
    meta = load_meta_raw(data_dir)
    counts = meta["OncotreePrimaryDisease"].value_counts()
    print(f"\n{'PrimaryDisease':<55} {'세포주 수':>8}")
    print("-" * 65)
    for disease, n in counts.items():
        print(f"  {disease:<53} {n:>8,}")


def main():
    parser = argparse.ArgumentParser(
        description="DepMap 26Q1 feature extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # CNV + 발현(z-score), TP53·KRAS·EGFR 유전자, Lung 세포주만 추출
  python extract.py --features cnv expr --genes TP53 KRAS EGFR --lineage Lung --output subset.csv

  # Lung + Breast의 LoF 변이 전체 추출
  python extract.py --features mut_lof --lineage Lung Breast --output lof.csv

  # 메타데이터 + 이진변이 (특정 세포주 지정)
  python extract.py --features meta mut --model-ids ACH-000001 ACH-000002 --output selected.csv

  # 사용 가능한 암종 계통 목록 보기
  python extract.py --list-lineages

사용 가능한 feature 이름:
  cnv        CNV log2 변환 (prefix: CNV_)
  expr       발현 z-score (prefix: EXP_)
  expr_tpm   발현 logTPM (prefix: EXP_)
  mut        변이 이진 행렬 HIGH+MODERATE (prefix: MUT_)
  mut_lof    LoF 변이 이진 행렬 (prefix: MUT_LoF_)
  mut_hot    Hotspot 변이 이진 행렬 (prefix: MUT_HOT_)
  meta       메타데이터 (one-hot 인코딩)
""",
    )
    parser.add_argument("--features", nargs="+", choices=list(FILE_MAP.keys()),
                        metavar="FEATURE",
                        help="추출할 피처 종류 (cnv expr expr_tpm mut mut_lof mut_hot meta)")
    parser.add_argument("--genes", nargs="+", metavar="GENE",
                        help="추출할 유전자 심볼 (생략 시 전체, meta에는 미적용)")
    parser.add_argument("--lineage", nargs="+", metavar="LINEAGE",
                        help="OncotreeLineage 필터 (예: Lung Breast)")
    parser.add_argument("--disease", nargs="+", metavar="DISEASE",
                        help="OncotreePrimaryDisease 필터")
    parser.add_argument("--model-ids", nargs="+", metavar="MODEL_ID",
                        help="특정 ModelID 지정 (예: ACH-000001)")
    parser.add_argument("--data-dir", default=DATA_DIR, metavar="DIR",
                        help=f"전처리 파일 디렉터리 (기본값: {DATA_DIR})")
    parser.add_argument("--output", "-o", default="extracted.csv", metavar="FILE",
                        help="출력 CSV 경로 (기본값: extracted.csv)")
    parser.add_argument("--list-lineages", action="store_true",
                        help="사용 가능한 암종 계통 목록 출력 후 종료")
    parser.add_argument("--list-diseases", action="store_true",
                        help="사용 가능한 PrimaryDisease 목록 출력 후 종료")

    args = parser.parse_args()

    if args.list_lineages:
        list_lineages(args.data_dir)
        return

    if args.list_diseases:
        list_diseases(args.data_dir)
        return

    if not args.features:
        parser.print_help()
        sys.exit("\n오류: --features 를 지정하세요.")

    # 샘플 필터링
    meta_raw = load_meta_raw(args.data_dir)
    samples = filter_samples(meta_raw, args.lineage, args.disease, args.model_ids)
    print(f"\n대상 세포주: {len(samples):,}개")

    # 피처 로드 및 병합
    blocks = []
    for feat in args.features:
        print(f"  로딩: {feat} ...", end="", flush=True)
        df = load_feature_block(args.data_dir, feat, args.genes, samples)
        print(f" → {df.shape[1]:,}개 피처")
        blocks.append(df)

    result = pd.concat(blocks, axis=1)

    # 중복 컬럼 제거 (expr + expr_tpm 동시 선택 등)
    result = result.loc[:, ~result.columns.duplicated()]

    print(f"\n최종 추출 행렬: {result.shape[0]:,} 세포주 × {result.shape[1]:,} 피처")
    print(f"저장 중: {args.output}")
    result.to_csv(args.output)
    print("완료.")


if __name__ == "__main__":
    main()
