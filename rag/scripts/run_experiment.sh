#!/bin/bash
# Usage: bash run_experiment.sh --k 10 --split practice --limit 350
# Example: bash run_experiment.sh --k 0 --split practice --limit 50

# 默认参数
K=10
SPLIT="practice"
LIMIT=350
DEVICE="cuda"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --k) K="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# 切换到项目根目录
cd "$(dirname "$0")/.." || exit 1

# 创建带时间戳的输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="rag/results/depth_k${K}_${SPLIT}_${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "============================================"
echo "Starting experiment: k=$K, split=$SPLIT, limit=$LIMIT"
echo "Output directory: $OUTDIR"
echo "============================================"

# 步骤1: 生成答案
echo ""
echo "[Step 1] Generating answers..."
python rag/scripts/generate_answers.py \
    --device "$DEVICE" \
    --split "$SPLIT" \
    --k "$K" \
    --output "$OUTDIR/sample_answers.jsonl" \
    ${LIMIT:+--limit $LIMIT}

if [ $? -ne 0 ]; then
    echo "ERROR: generate_answers.py failed"
    exit 1
fi
echo "Answers saved to $OUTDIR/sample_answers.jsonl"

# 步骤2: 清洗特殊字符（避免评估时报错）
echo ""
echo "[Step 2] Cleaning special characters..."
python -c "
import json
infile = '$OUTDIR/sample_answers.jsonl'
outfile = '$OUTDIR/sample_answers_cleaned.jsonl'
def clean(t):
    return ''.join(c for c in t if ord(c) >= 32 or c in '\n\r\t')
with open(infile, 'r', encoding='utf-8') as fi, \
     open(outfile, 'w', encoding='utf-8') as fo:
    for line in fi:
        d = json.loads(line)
        d['transcript'] = clean(d.get('transcript', ''))
        d['answer'] = clean(d.get('answer', ''))
        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
print('Cleaning done.')
"
echo "Cleaned file: $OUTDIR/sample_answers_cleaned.jsonl"

# 步骤3: 评估答案
echo ""
echo "[Step 3] Evaluating answers..."
python rag/scripts/evaluate_answers.py \
    --split "$SPLIT" \
    --candidates "$OUTDIR/sample_answers_cleaned.jsonl" \
    --out-dir "$OUTDIR/judge_results" \
    --allow-subset

if [ $? -ne 0 ]; then
    echo "WARNING: evaluate_answers.py had some errors (maybe some queries still fail)"
fi

# 步骤4: 显示结果摘要
echo ""
echo "============================================"
echo "Experiment complete!"
echo "Results summary:"
if [ -f "$OUTDIR/judge_results/answer_judge_summary.json" ]; then
    cat "$OUTDIR/judge_results/answer_judge_summary.json"
else
    echo "Summary file not found, check $OUTDIR/judge_results/"
fi
echo "============================================"
echo "All outputs are in: $OUTDIR"