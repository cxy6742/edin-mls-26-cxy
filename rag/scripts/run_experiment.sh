/bin/bash
# Usage: bash run_experiment.sh --k 10 --split practice --limit 350 --runs 3

# 默认参数
K=10
SPLIT="practice"
LIMIT=350
DEVICE="cuda"
RUNS=1

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --k) K="$2"; shift ;;
        --split) SPLIT="$2"; shift ;;
        --limit) LIMIT="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        --runs) RUNS="$2"; shift ;;
    esac
    shift
done

cd "$(dirname "$0")/.." || exit 1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="rag/results/depth_k${K}_${SPLIT}_${TIMESTAMP}"
mkdir -p "$OUTDIR"

echo "============================================"
echo "Starting experiment: k=$K, split=$SPLIT, limit=$LIMIT, runs=$RUNS"
echo "Output directory: $OUTDIR"
echo "============================================"

# 循环运行多次取平均
for (( i=1; i<=$RUNS; i++ ))
do
   echo ""
   echo "--- [Run $i/$RUNS] ---"
   
   # 步骤1: 生成答案
   python rag/scripts/generate_answers.py \
       --device "$DEVICE" \
       --split "$SPLIT" \
       --k "$K" \
       --output "$OUTDIR/run_${i}_answers.jsonl" \
       ${LIMIT:+--limit $LIMIT}

   # 步骤2: 清洗特殊字符
   python -c "
import json
infile = '$OUTDIR/run_${i}_answers.jsonl'
outfile = '$OUTDIR/run_${i}_answers_cleaned.jsonl'
def clean(t):
    return ''.join(c for c in t if ord(c) >= 32 or c in '\n\r\t')
with open(infile, 'r', encoding='utf-8') as fi, open(outfile, 'w', encoding='utf-8') as fo:
    for line in fi:
        d = json.loads(line)
        d['transcript'] = clean(d.get('transcript', ''))
        d['answer'] = clean(d.get('answer', ''))
        fo.write(json.dumps(d, ensure_ascii=False) + '\n')
"

   # 步骤3: 评估答案
   python rag/scripts/evaluate_answers.py \
       --split "$SPLIT" \
       --candidates "$OUTDIR/run_${i}_answers_cleaned.jsonl" \
       --out-dir "$OUTDIR/run_${i}_judge_results" \
       --allow-subset
done

# 步骤4: 汇总多次运行结果（需自行完善汇总脚本，此处为示意）
echo ""
echo "============================================"
echo "All runs complete! Calculating average..."
# python rag/scripts/aggregate_results.py --dir $OUTDIR --runs $RUNS
echo "Results saved in: $OUTDIR"