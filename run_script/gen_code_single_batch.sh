#!/usr/bin/env bash
#
# run_all_configurable.sh — dispatch specified number of jobs (i=0..N-1)
# across specified GPUs in parallel, running each GPU’s jobs sequentially.

# Default values
DEFAULT_GPU_IDS_VAL="0"
DEFAULT_MODEL_PATH_VAL="/data/genai/models/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_NUM_JOBS_VAL=20 # Corresponds to jobs 0-19

# --- Usage Function ---
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Dispatch i=0..N-1 jobs across specified GPUs in parallel."
    echo ""
    echo "Options:"
    echo "  --gpus \"ID1,ID2,...\"  Specify comma-separated GPU IDs to use (e.g., \"0,1,3\")."
    echo "                           Overrides ENV_GPU_IDS. Default: \"${DEFAULT_GPU_IDS_VAL}\"."
    echo "  --model MODEL_PATH     Specify the model path."
    echo "                           Overrides ENV_MODEL_PATH. Default: \"${DEFAULT_MODEL_PATH_VAL}\"."
    echo "  --common-args \"ARGS\"   Specify all common arguments for main.py (e.g., \"-hb 32 --temperature 0.8\")."
    echo "                           Overrides ENV_COMMON_ARGS. If not set, default args will be used."
    echo "                           If you set this, ensure you include -im if needed."
    echo "  --num-jobs N           Specify the total number of jobs to launch (e.g., 20 for jobs 0-19)."
    echo "                           Overrides ENV_NUM_JOBS. Default: ${DEFAULT_NUM_JOBS_VAL}."
    echo "  -h, --help             Display this help message."
    echo ""
    echo "Environment Variables (used if corresponding command-line options are not set):"
    echo "  ENV_GPU_IDS:           Comma-separated GPU IDs (e.g., \"0,1,3\")."
    echo "  ENV_MODEL_PATH:        Path to the model."
    echo "  ENV_COMMON_ARGS:       String of common arguments for main.py."
    echo "  ENV_NUM_JOBS:          Total number of jobs."
    exit 1
}

# --- Parse Command-Line Arguments ---
REQUESTED_GPU_IDS_STR=""
REQUESTED_MODEL_PATH=""
REQUESTED_COMMON_ARGS_STR=""
REQUESTED_NUM_JOBS=""

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpus)
        REQUESTED_GPU_IDS_STR="$2"
        shift 2 # past argument and value
        ;;
        --model)
        REQUESTED_MODEL_PATH="$2"
        shift 2 # past argument and value
        ;;
        --common-args)
        REQUESTED_COMMON_ARGS_STR="$2"
        shift 2 # past argument and value
        ;;
        --num-jobs)
        REQUESTED_NUM_JOBS="$2"
        shift 2 # past argument and value
        ;;
        -h|--help)
        usage
        ;;
        *)    # unknown option
        echo "Error: Unknown option: $1"
        usage
        ;;
    esac
done

# --- Determine Final Configurations ---

# 1) GPU IDs
FINAL_GPU_IDS_STR=""
if [[ -n "$REQUESTED_GPU_IDS_STR" ]]; then
    FINAL_GPU_IDS_STR="$REQUESTED_GPU_IDS_STR"
    echo "Using GPU IDs from --gpus: ${FINAL_GPU_IDS_STR}"
elif [[ -n "$ENV_GPU_IDS" ]]; then
    FINAL_GPU_IDS_STR="$ENV_GPU_IDS"
    echo "Using GPU IDs from ENV_GPU_IDS: ${FINAL_GPU_IDS_STR}"
else
    FINAL_GPU_IDS_STR="$DEFAULT_GPU_IDS_VAL"
    echo "Using default GPU IDs: ${FINAL_GPU_IDS_STR}"
fi

if [[ -z "$FINAL_GPU_IDS_STR" ]]; then
  echo "Error: GPU IDs string is empty. Please specify GPU IDs."
  usage
fi

IFS=',' read -r -a TEMP_GPU_ID_LIST <<< "$FINAL_GPU_IDS_STR"
GPU_ID_LIST=()
for id_val in "${TEMP_GPU_ID_LIST[@]}"; do # Renamed 'id' to 'id_val' to avoid conflict if user has 'id' command
  if [[ -n "$id_val" ]]; then # Filter out empty IDs that might result from "val1,,val2"
    GPU_ID_LIST+=("$id_val")
  fi
done

NUM_SPECIFIED_GPUS=${#GPU_ID_LIST[@]}
if [[ $NUM_SPECIFIED_GPUS -eq 0 ]]; then
  echo "Error: No valid GPU IDs found after parsing. Original string: '${FINAL_GPU_IDS_STR}'"
  usage
fi
echo "Effective GPU IDs to use: ${GPU_ID_LIST[*]}"
echo "Number of GPUs for parallel execution: ${NUM_SPECIFIED_GPUS}"

# 2) Model Path
FINAL_MODEL_PATH=""
if [[ -n "$REQUESTED_MODEL_PATH" ]]; then
    FINAL_MODEL_PATH="$REQUESTED_MODEL_PATH"
    echo "Using Model Path from --model: ${FINAL_MODEL_PATH}"
elif [[ -n "$ENV_MODEL_PATH" ]]; then
    FINAL_MODEL_PATH="$ENV_MODEL_PATH"
    echo "Using Model Path from ENV_MODEL_PATH: ${FINAL_MODEL_PATH}"
else
    FINAL_MODEL_PATH="$DEFAULT_MODEL_PATH_VAL"
    echo "Using default Model Path: ${FINAL_MODEL_PATH}"
fi

# 3) Common Arguments
FINAL_COMMON_ARGS=""
if [[ -n "$REQUESTED_COMMON_ARGS_STR" ]]; then
    FINAL_COMMON_ARGS="$REQUESTED_COMMON_ARGS_STR"
    echo "Using Common Args from --common-args: ${FINAL_COMMON_ARGS}"
elif [[ -n "$ENV_COMMON_ARGS" ]]; then
    FINAL_COMMON_ARGS="$ENV_COMMON_ARGS"
    echo "Using Common Args from ENV_COMMON_ARGS: ${FINAL_COMMON_ARGS}"
else
    # Construct default common args, ensuring model path is quoted for spaces
    FINAL_COMMON_ARGS="-im \"${FINAL_MODEL_PATH}\" \
-d evaluation.jsonl -mp False -np 1 -ip 0 \
-ft -hb 16 --temperature 0.2"
    echo "Using default Common Args, constructed with Model Path: ${FINAL_MODEL_PATH}"
fi

# 4) Number of Jobs
FINAL_NUM_JOBS=""
if [[ -n "$REQUESTED_NUM_JOBS" ]]; then
    FINAL_NUM_JOBS="$REQUESTED_NUM_JOBS"
    echo "Using Number of Jobs from --num-jobs: ${FINAL_NUM_JOBS}"
elif [[ -n "$ENV_NUM_JOBS" ]]; then
    FINAL_NUM_JOBS="$ENV_NUM_JOBS"
    echo "Using Number of Jobs from ENV_NUM_JOBS: ${FINAL_NUM_JOBS}"
else
    FINAL_NUM_JOBS="$DEFAULT_NUM_JOBS_VAL"
    echo "Using default Number of Jobs: ${FINAL_NUM_JOBS}"
fi

# Validate FINAL_NUM_JOBS
if ! [[ "$FINAL_NUM_JOBS" =~ ^[0-9]+$ ]] || [[ "$FINAL_NUM_JOBS" -lt 1 ]]; then
    echo "Error: Number of jobs must be a positive integer (>= 1). Got: '${FINAL_NUM_JOBS}'"
    usage
fi

MAX_JOB_INDEX=$((FINAL_NUM_JOBS-1))
echo "Total jobs to run: ${FINAL_NUM_JOBS} (indices 0 to ${MAX_JOB_INDEX})"


# --- Launch one backgrounded subshell per specified GPU ---
# The loop iterates using an index for the GPU_ID_LIST
for gpu_idx in $(seq 0 $((NUM_SPECIFIED_GPUS-1))); do
  ACTUAL_GPU_ID=${GPU_ID_LIST[gpu_idx]}
  (
    # for each i that maps to this GPU (job_index % NUM_SPECIFIED_GPUS == gpu_idx)
    # Jobs run from 0 to MAX_JOB_INDEX
    for i in $(seq ${gpu_idx} ${NUM_SPECIFIED_GPUS} ${MAX_JOB_INDEX}); do
      echo "GPU_ID ${ACTUAL_GPU_ID} (assigned task index ${gpu_idx}): running job i=${i}"
      
      COMMAND_TO_RUN="python3 main.py GEN_HDL ${FINAL_COMMON_ARGS} -i ${i}"
      
      echo "Executing on GPU ${ACTUAL_GPU_ID}: CUDA_VISIBLE_DEVICES=${ACTUAL_GPU_ID} ${COMMAND_TO_RUN}"
      CUDA_VISIBLE_DEVICES=${ACTUAL_GPU_ID} eval "${COMMAND_TO_RUN}"
    done
  ) &
done

# --- Wait for all to finish ---
wait
echo "✅ All jobs (i=0..${MAX_JOB_INDEX}) complete on GPUs: ${GPU_ID_LIST[*]}."