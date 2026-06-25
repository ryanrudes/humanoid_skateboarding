SCRIPT_DIR=$(dirname $(realpath $0))

# Visualize the reference target poses (dataset/ref_pose/*.npy) as orbiting
# videos in video/ref_pose/. All args are forwarded to view_ref_pose.py
# (run with --help to see options), e.g.
#       bash test_scene/view_ref_pose.sh --still
#       bash test_scene/view_ref_pose.sh --data dataset/ref_pose/steer_start_pose_b.npy

uv run python ${SCRIPT_DIR}/view_ref_pose.py "$@"
