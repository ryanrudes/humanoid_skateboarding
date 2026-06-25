SCRIPT_DIR=$(dirname $(realpath $0))

# Replay the mocap dataset in plain MuJoCo and render to video/dataset_replay/.
# All args are forwarded to replay_dataset.py (run with --help to see options),
# e.g.  bash test_scene/replay.sh --data dataset/skate_push/human_push_1.npy
#       bash test_scene/replay.sh --camera fixed --width 1280 --height 720

uv run python ${SCRIPT_DIR}/replay_dataset.py "$@"
