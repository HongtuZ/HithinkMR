import joblib


def load_robot_motion(motion_file):
    """
    Load robot motion data from a pickle file.
    """
    motion_data = joblib.load(motion_file)
    motion_fps = motion_data["fps"]
    motion_root_pos = motion_data["root_pos_w"]
    motion_root_rot = motion_data["root_quat_w"]  # wxyz
    motion_dof_pos = motion_data["joint_pos"]
    motion_local_body_pos = motion_data["body_pos_b"]
    motion_link_body_list = motion_data["body_names"]
    return (
        motion_data,
        motion_fps,
        motion_root_pos,
        motion_root_rot,
        motion_dof_pos,
        motion_local_body_pos,
        motion_link_body_list,
    )
