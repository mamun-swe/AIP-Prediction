#!/usr/bin/env python3
"""
Script 01 — Data Preparation
==============================
This script:
  1. Generates a sample dataset for testing (if real data is not yet available)
  2. Alternatively, loads the Manavalan 2018 benchmark from CSV/FASTA
  3. Cleans and filters sequences
  4. Removes redundancy (CD-HIT equivalent)
  5. Saves processed dataset to data/processed/dataset.csv

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO GET THE REAL DATASET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Option A — Manavalan 2018 Benchmark (Recommended):
  1. Go to: http://www.thegleelab.org/AIPpred
  2. Download the benchmark dataset from the "Dataset" section
  3. It contains 1,678 AIPs and 2,516 non-AIPs
  4. Save as:
       data/raw/aip_positive.fasta   ← AIP sequences
       data/raw/aip_negative.fasta   ← non-AIP sequences
  5. Run: python scripts/01_prepare_data.py --mode fasta

Option B — From CSV:
  1. Prepare a CSV with columns: sequence, label (1=AIP, 0=non-AIP)
  2. Save as data/raw/dataset.csv
  3. Run: python scripts/01_prepare_data.py --mode csv

Option C — Sample data (for testing the pipeline):
  Run: python scripts/01_prepare_data.py --mode sample

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.preprocessor import (
    build_dataset_from_fastas,
    build_dataset_from_csv,
    clean_dataframe,
    remove_redundancy_python,
)
from src.data.splitter import split_and_save
from src.utils.logger import get_logger

log = get_logger("01_prepare_data", log_file="results/logs/01_prepare_data.log")


# ─── sample dataset for pipeline testing ───────────────────────────────────
# These are real AIP sequences from published literature used as examples.
# Do NOT use this as your actual training data — get the full Manavalan 2018
# benchmark from the AIPpred server.

SAMPLE_POSITIVES = [
    "GILDTAGLNLYVF", "FLPILASLAAKFGPK", "MKGAVFSGF", "KFLHSAGKFGKALG",
    "RLCRIVVIRVCR", "GIMDTAGLNLYVF", "RSLRKSDFY", "RYLNKRYLNKRYLNK",
    "PVCSAYDQMREPVHF", "KVFGRCELAA", "FNMQCQRRFY", "RFGRCVKRK",
    "IEPRLKRTRL", "SIFGAIAGFLE", "GVSIFGAIAGFLEG", "RFGSAVRSAGRK",
    "DCYCLKSF", "MKGAVFSGFKAVGK", "GLFDAIKKVASVIGGL", "RLVTAFATYL",
    "RLARIVVIRVCR", "CRTPVKRL", "AVLKKVLTTGLPALI", "GIGAVLKVLTTGLPALI",
    "FFGCAPGQK", "CSIFGAIAGFLE", "KLAKLAKKLAKLAK", "KLAKLAKLAKLAKLAK",
    "KLAKLAK", "GRNSFRY", "RRLSYSRRRF", "YLYEIARRHPYFYAPELLF",
    "ACDEFGHIKLM", "MNPQRSTVWY", "ACMNPQRSTVWY", "KLMNPQRS",
    "LKLKLKLKL", "KWKLFKK", "RWKIFKK", "RRWWRF",
    "ILPWKWPWWPWRR", "GLFDAIKKVASVIGGLSALKPK", "GIGKFLHSAGKFGKAFVGEIMKS",
    "KWKLFKKIEKVGQNIRDGIIKAG", "ACDEFGHIKLMNPQRSTVWY",
    "FLPILASLAAKFG", "MKGAVFS", "RSLRKSD", "RLCRIVVIR",
    "DCYCLK", "MKGAVFSGFK", "FNMQCQRR", "RFGRCVK",
]

SAMPLE_NEGATIVES = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL",
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
    "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV",
    "MSSRSVSSSGYKLLDLQELAKVSQEEQKLISEEDL",
    "MATGSRTSLLLAFGLLCLPWLQEGSAFPTEPPVGSSVASSVPSQKTYPGSSSGAPPPSGGSTQQSTAPVSQKDNPQPK",
    "MKTIIALSYIFCLVFA",  "MKCLLLALALTCGAQALIVTQSPGTLSLSPGERATLSCRASQSVGSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPWTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC",
    "ACNLYVQCPAGISMLPAGPAGPVPGGAMR",
    "QVQLQQSGAELAKPGASVKMSCKASGYTFTSYWMHWVKQRPGQGLEWIGYINPSRGYTNYNQKFKDKATLTTDKSSSTAYMQLSSLTSEDSAVYYCAR",
    "GASGAGASGAGASGAGA", "PAPAPAPAPAPAPA", "STSTSTSTSTS",
    "QQQQQQQQQQQQ", "EEEEEEEEEEEE", "NNNNNNNNNNNN",
    "RRRRRRRRRRRR", "KKKKKKKKKKKK", "DDDDDDDDDDDD",
    "GGGGGGGGGGGG", "AAAAAAAAAAAA", "LLLLLLLLLLLL",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCAR",
    "SYSMEHFRWGKPVGKKRRPVKVYPNGAEDESAEAFPLEF",
    "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTGDLYELLNELQKDFEDNYSNSGK",
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQGVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT",
    "MALSQFGDGDLSSISGPHQIRSKLPPRSAPFKQLNKPRLPPSYAPVVAFTTQATPSTAEAPIKSAAGAIAAFLAIAALFVKADTNIAQKFRTLQALREVTEEPELRSSMKWVPNTMSVNGQAGDILLEQLTHEKYGAPAMLRHLIQNLPIGSKLPLTGGPHTILHPFLGPILPASRGERLALGIAQRALSTLKDRNAGKIAQGLKSGPSFGGRFSGQGKMLPKSVMDIKIVKTAKEAVGWRPNVPVDLSTQCPLTIAEPEPNLYVTPYEWSGFSAGGFPASQLKEPIMHTSGAAAKDATKTDLSAAGAQGTATITSLAQASRLHREQKKLSSEQPELLSRKRAAESATAESAAAPTAEADQEKAKRETAEGQLSEVKHYQEGKTLPQHLQTLLTAPAPEPSKNLASEKFPQPKNLVDKSLLEESRSGNDRPLHSKSTIDVAEATLSSPPLRSPAAPPEAPNSNTSSLSSAASPSLQPKGPDPLALPPPSSPKLEALSQPKRPNKNISLPVSMDPSSSIAPADATSSPLEKSSKLGSLLSALQPTTLSASTAKAKDAMSDATSNIGLSREQLLDEALANLLKRDAGALKQMEEKASQALTDELTHLMQAIAKKSEKIGQFLTSQLKELKIASGLTTAFLDAQASPVAKDAQRGAEKEKAEIAAQAKQKAEAQKRSAGEKAKQEIEDSIKARLKQKQEMIKAIKQAEKKLAETLKAQEFAEAKKK",
    "KVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "ACDEFGHIKLMNPQRSTVWYACDE", "GHIKLMNPQRSTVWYACDE",
    "LMNPQRSTVWYACDEFGHI", "QRSTVWYACDEFGHIKLMN",
]

# ─── main ────────────────────────────────────────────────────────────────────

def create_sample_dataset(output_path: str, n_pos: int = 50, n_neg: int = 80):
    """Create a small sample dataset for pipeline testing."""
    pos = pd.DataFrame({
        "sequence": SAMPLE_POSITIVES[:n_pos],
        "label": 1
    })
    neg = pd.DataFrame({
        "sequence": [s[:40] for s in SAMPLE_NEGATIVES[:n_neg]],
        "label": 0
    })
    df = pd.concat([pos, neg], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Sample dataset saved → {output_path} "
             f"({len(pos)} pos, {len(neg)} neg)")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="AIP dataset preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--mode", choices=["fasta", "csv", "sample"],
                        default="sample",
                        help="Data loading mode (default: sample)")
    parser.add_argument("--pos_fasta", default="data/raw/aip_positive.fasta",
                        help="Path to positive FASTA (mode=fasta)")
    parser.add_argument("--neg_fasta", default="data/raw/aip_negative.fasta",
                        help="Path to negative FASTA (mode=fasta)")
    parser.add_argument("--csv", default="data/raw/dataset.csv",
                        help="Path to input CSV (mode=csv)")
    parser.add_argument("--seq_col", default="sequence")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--cdhit_threshold", type=float, default=0.8)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    Path("results/logs").mkdir(parents=True, exist_ok=True)

    processed_csv = "data/processed/dataset.csv"

    if args.mode == "sample":
        log.info("Mode: sample — generating test dataset...")
        df = create_sample_dataset(processed_csv)

    elif args.mode == "fasta":
        log.info("Mode: fasta — loading from FASTA files...")
        assert Path(args.pos_fasta).exists(), \
            f"Positive FASTA not found: {args.pos_fasta}"
        assert Path(args.neg_fasta).exists(), \
            f"Negative FASTA not found: {args.neg_fasta}"
        df = build_dataset_from_fastas(
            pos_fasta=args.pos_fasta,
            neg_fasta=args.neg_fasta,
            min_len=args.min_len,
            max_len=args.max_len,
            redundancy_threshold=args.cdhit_threshold,
            output_csv=processed_csv
        )

    elif args.mode == "csv":
        log.info("Mode: csv — loading from CSV...")
        assert Path(args.csv).exists(), f"CSV not found: {args.csv}"
        df = build_dataset_from_csv(
            csv_path=args.csv,
            seq_col=args.seq_col,
            label_col=args.label_col,
            min_len=args.min_len,
            max_len=args.max_len,
            redundancy_threshold=args.cdhit_threshold,
            output_csv=processed_csv
        )

    # Split
    log.info("Splitting into train/test...")
    train_df, test_df = split_and_save(
        df, splits_dir="data/splits",
        train_ratio=args.train_ratio,
        random_state=args.random_state
    )

    log.info("━" * 50)
    log.info("Dataset preparation complete.")
    log.info(f"  Processed : {processed_csv}")
    log.info(f"  Train     : data/splits/train.csv ({len(train_df)} samples)")
    log.info(f"  Test      : data/splits/test.csv  ({len(test_df)} samples)")
    log.info("━" * 50)
    log.info("Next step: python scripts/02_extract_features.py")


if __name__ == "__main__":
    main()
