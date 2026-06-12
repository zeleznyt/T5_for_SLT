import os

from utils.keypoint_dataset import KeypointDatasetJSON


POSE_NORMALIZATION_ORDER = (
    "global-pose_landmarks",
    "local-right_hand_landmarks",
    "local-left_hand_landmarks",
    "local-face_landmarks",
)


def resolve_data_path(data_dir, path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def get_pose_json_dir(sign_data_args, split):
    pose_config = sign_data_args['visual_features']['pose']
    normalization_config = pose_config.get('normalization') or {}
    json_dir_keys = {
        'train': ('train_json_dir',),
        'dev': ('dev_json_dir', 'val_json_dir'),
        'test': ('test_json_dir',),
    }

    for key in json_dir_keys[split]:
        json_dir = normalization_config.get(key)
        if json_dir is not None:
            return resolve_data_path(sign_data_args['data_dir'], json_dir)
    return None


def build_pose_json_dataset(sign_data_args, split, augmentation_configs=None):
    pose_config = sign_data_args['visual_features']['pose']
    normalization_config = pose_config.get('normalization') or {}
    json_dir = get_pose_json_dir(sign_data_args, split)

    if json_dir is None:
        print(f'Raw pose JSON directory is not configured for {split}; using pose metadata/H5 if available.')
        return None

    if not os.path.isdir(json_dir):
        print(f'Raw poses not found in {json_dir}; using pose metadata/H5 if available.')
        return None

    pose_dataset = KeypointDatasetJSON(json_folder=json_dir,
                                       kp_normalization=POSE_NORMALIZATION_ORDER,
                                       kp_normalization_method=normalization_config.get('normalization_method', 'sign_space'),
                                       data_key=normalization_config.get('data_key', 'cropped_keypoints'),
                                       missing_values=pose_config.get('missing_values'),
                                       augmentation_configs=augmentation_configs or [],
                                       interpolate=pose_config.get('interpolate', -1),
                                       )
    print(f'{split.capitalize()} raw pose data path: {json_dir}')
    return pose_dataset
