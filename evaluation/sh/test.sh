set -ex

export CUDA_VISIBLE_DEVICES="0"
PROMPT_TYPE="qwen25-math-cot"

# MODEL_NAME_OR_PATH="Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_NAME_OR_PATH="/lamda12/yangxw/codes/OpenRLHF_prm/examples/checkpoint/qwen-1b-prm"
# 替换成对应checkpoint

OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval

SPLIT="test"
NUM_TEST_SAMPLE=-1


# English competition datasets
DATA_NAME="aime24,amc23"
# DATA_NAME="amc23"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.9 \
    --n_sampling 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \