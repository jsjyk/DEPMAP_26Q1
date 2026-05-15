"""
MSigDB / KEGG pathway 데이터 다운로드 스크립트.
결과물: data/pathways/*.gmt

방법 A (권장): gseapy 사용
  pip install gseapy
  python script/download_pathways.py

방법 B: MSigDB 수동 다운로드
  https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp
  → h.all.*.symbols.gmt  (Hallmark)
  → c2.cp.kegg.*.symbols.gmt  (KEGG)
  파일을 data/pathways/ 에 위치시킨 후 PathwayDB.from_gmt() 사용

실행:
  python script/download_pathways.py
  python script/download_pathways.py --sets hallmark kegg reactome
"""

import argparse
import os

OUT_DIR = "data/pathways"

# gseapy에서 사용 가능한 gene set 이름
GENE_SET_MAP = {
    "hallmark":  "MSigDB_Hallmark_2020",
    "kegg":      "KEGG_2021_Human",
    "reactome":  "Reactome_2022",
    "wikipathways": "WikiPathways_2021_Human",
}


def download_via_gseapy(sets: list[str], out_dir: str) -> None:
    try:
        import gseapy as gp
    except ImportError:
        print("gseapy가 설치되어 있지 않습니다. pip install gseapy")
        print("또는 MSigDB에서 수동으로 .gmt 파일을 다운로드하세요.")
        print("  → https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp")
        return

    os.makedirs(out_dir, exist_ok=True)
    for s in sets:
        gs_name = GENE_SET_MAP.get(s, s)
        out_path = os.path.join(out_dir, f"{s}.gmt")
        if os.path.exists(out_path):
            print(f"  이미 존재: {out_path}")
            continue
        print(f"  다운로드: {gs_name} → {out_path}")
        try:
            library = gp.get_library(name=gs_name, organism="Human")
            with open(out_path, "w") as f:
                for name, genes in library.items():
                    f.write(name + "\tna\t" + "\t".join(genes) + "\n")
            print(f"  저장 완료: {len(library)}개 pathway")
        except Exception as e:
            print(f"  실패: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pathway 데이터 다운로드")
    parser.add_argument(
        "--sets", nargs="+",
        default=["hallmark", "kegg"],
        choices=list(GENE_SET_MAP.keys()),
        help="다운로드할 gene set (기본: hallmark kegg)",
    )
    parser.add_argument("--out", default=OUT_DIR, help="출력 디렉터리")
    args = parser.parse_args()

    print("=== Pathway 다운로드 ===")
    download_via_gseapy(args.sets, args.out)

    print(f"\n완료. PathwayDB 로드 방법:")
    for s in args.sets:
        print(f"  db = PathwayDB.from_gmt('{args.out}/{s}.gmt', vocab)")
    print(f"  또는 db = PathwayDB.from_gseapy('MSigDB_Hallmark_2020', vocab)")


if __name__ == "__main__":
    main()
