SCRIPT_DIR=$(dirname $(realpath $0))

if [ -z "$1" ]; then
  echo "Usage: $0 path_to_policy.onnx [extra sim_no_board.py args]"
  echo "  e.g. $0 ckpts/test.onnx --init_pose default --v0 0.5"
  exit 1
fi

ckpt_path=$1
shift

uv run python test_scene/sim_no_board.py \
    --xml ${SCRIPT_DIR}/mjlab_scene.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --policy_frequency 50 \
    "$@"
