"""
STRING DB v12.0 (Human) 다운로드 및 전처리 스크립트.
결과물: data/ppi/ppi_edges.csv (gene_a, gene_b, score)

실행:
  python script/download_ppi.py
  python script/download_ppi.py --min-score 0.7 --out data/ppi/ppi_edges.csv
"""

import argparse
import gzip
import os
import urllib.request

URLS = {
    "links": "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz",
    "info":  "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
}


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  이미 존재, 건너뜀: {dest}")
        return
    print(f"  다운로드: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  저장: {dest}")


def process(
    links_gz: str,
    info_gz: str,
    out_csv: str,
    min_score: float = 0.7,
) -> None:
    import pandas as pd

    print("\n[1/3] 단백질 정보 로딩 (gene symbol 매핑)...")
    with gzip.open(info_gz, "rt") as f:
        info = pd.read_csv(f, sep="\t", usecols=["#string_protein_id", "preferred_name"])
    id2gene: dict[str, str] = dict(
        zip(info["#string_protein_id"], info["preferred_name"])
    )
    print(f"  {len(id2gene):,}개 단백질")

    print("[2/3] 상호작용 엣지 로딩 및 필터링...")
    with gzip.open(links_gz, "rt") as f:
        links = pd.read_csv(f, sep=" ")

    min_score_int = int(min_score * 1000)
    links = links[links["combined_score"] >= min_score_int].copy()
    print(f"  combined_score ≥ {min_score_int}: {len(links):,}개 엣지 남음")

    print("[3/3] gene symbol 변환 및 저장...")
    links["gene_a"] = links["protein1"].map(id2gene)
    links["gene_b"] = links["protein2"].map(id2gene)
    links["score"]  = links["combined_score"] / 1000.0
    links = links.dropna(subset=["gene_a", "gene_b"])

    out = links[["gene_a", "gene_b", "score"]]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"  저장 완료: {out_csv}  ({len(out):,}개 엣지)")


def main() -> None:
    parser = argparse.ArgumentParser(description="STRING DB PPI 다운로드 및 전처리")
    parser.add_argument("--min-score", type=float, default=0.7,
                        help="최소 combined_score (0–1, 기본값 0.7 = high confidence)")
    parser.add_argument("--out", default="data/ppi/ppi_edges.csv",
                        help="출력 CSV 경로")
    parser.add_argument("--cache-dir", default="data/ppi/raw",
                        help="다운로드 캐시 디렉터리")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    links_gz = os.path.join(args.cache_dir, "9606.protein.links.v12.0.txt.gz")
    info_gz  = os.path.join(args.cache_dir, "9606.protein.info.v12.0.txt.gz")

    print("=== STRING DB 다운로드 ===")
    download(URLS["links"], links_gz)
    download(URLS["info"],  info_gz)

    print("\n=== 전처리 ===")
    process(links_gz, info_gz, args.out, args.min_score)

    print(f"\n완료. PPIGraph 로드 방법:")
    print(f"  from src.hmot import GeneVocab")
    print(f"  from src.hmot.ppi import PPIGraph")
    print(f"  vocab = GeneVocab.from_preprocessed('DepMap_26Q1_preprocessed')")
    print(f"  ppi   = PPIGraph.from_csv('{args.out}', vocab)")


if __name__ == "__main__":
    main()
