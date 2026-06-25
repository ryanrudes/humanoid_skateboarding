SCRIPT_DIR=$(dirname $(realpath $0))

if [ -z "$1" ]; then
  echo "Usage: $0 path_to_policy.onnx [robot|xml]"
  echo "  robot: g1 (default) or x2; or pass an explicit scene .xml path"
  exit 1
fi

ckpt_path=$1
robot=${2:-g1}

case "$robot" in
  g1) xml=${SCRIPT_DIR}/mjlab_scene.xml ;;
  x2) xml=${SCRIPT_DIR}/mjlab_scene_x2.xml ;;
  *)  xml=$robot ;;  # treat the argument as an explicit scene XML path
esac

uv run python test_scene/sim.py \
    --xml ${xml} \
    --policy ${ckpt_path} \
    --device cuda \
    --policy_frequency 50
