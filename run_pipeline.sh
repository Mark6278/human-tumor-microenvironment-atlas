#!/usr/bin/env bash
set -e

echo "=================================================================="
echo "  Universal Human Tumor Microenvironment & Multi-Modal Atlas"
echo "=================================================================="

# If the user passed any CLI arguments (e.g. ./run_pipeline.sh --input data/file.h5 --genes EGFR MAP2),
# forward all arguments directly to main.py!
if [ "$#" -gt 0 ]; then
    echo -e "\n[CLI Mode] Running..."
    python main.py "$@"
    exit 0
fi

# Otherwise, run the full default batch workflow (Demos + all local files in data/)
echo -e "\n[Batch Mode] running full demo"

echo -e "\n[1/3] Running Built-in 10x Human Breast Cancer Visium Demo..."
python main.py

echo -e "\n[2/3] Running Built-in Multi-Donor Human PBMC Cohort Demo..."
python main.py --cohort-demo

if [ -d "data" ]; then
    echo -e "\n[3/3] Scanning 'data/' for local 10x Genomics & spatial datasets..."
    shopt -s nullglob
    for file in data/*.h5 data/*.h5ad data/*.parquet data/*.tif data/*.tiff data/*.csv; do
        if [[ "$file" == *imagealignment* ]]; then
            echo "  -> Skipping alignment matrix: $file"
            continue
        fi
        echo "  -> Processing: $file"
        python main.py --input "$file"
    done
    shopt -u nullglob
fi

echo -e "\nAll pipelines completed! Figures saved in 'figures/' and tables in 'results/'."
