heavy = [
        {'name': 'rotate', 'angle': (-6, 6), 'p': 1.0},
        {'name': 'shear', 'angle_x': (-6, 6), 'angle_y': (-6, 6), 'p': 0.75},
        {'name': 'perspective', 'portion': (-0.15, 0.15), 'reference_size': 512, 'p': 0.5},
        {'name': 'rotate_hand', 'angle': (-10, 10), 'rotation_center': 'shoulder', 'p': 0.75},
        {'name': 'rotate_hand', 'angle': (-10, 10), 'rotation_center': 'elbow', 'p': 0.75},
        {'name': 'rotate_hand', 'angle': (-10, 10), 'rotation_center': 'wrist', 'p': 0.75},
        {'name': 'noise', 'std': 1.5, 'p': 1.0}
]
medium = [
        {'name': 'rotate', 'angle': (-4.5, 4.5), 'p': 0.75},
        {'name': 'shear', 'angle_x': (-4.5, 4.5), 'angle_y': (-4.5, 4.5), 'p': 0.56},
        {'name': 'perspective', 'portion': (-0.11, 0.11), 'reference_size': 512, 'p': 0.38},
        {'name': 'rotate_hand', 'angle': (-7.5, 7.5), 'rotation_center': 'shoulder', 'p': 0.56},
        {'name': 'rotate_hand', 'angle': (-7.5, 7.5), 'rotation_center': 'elbow', 'p': 0.56},
        {'name': 'rotate_hand', 'angle': (-7.5, 7.5), 'rotation_center': 'wrist', 'p': 0.56},
        {'name': 'noise', 'std': 1.5, 'p': 0.75}
]
light = [
        {'name': 'rotate', 'angle': (-3.0, 3.0), 'p': 0.5},
        {'name': 'shear', 'angle_x': (-3.0, 3.0), 'angle_y': (-3.0, 3.0), 'p': 0.38},
        {'name': 'perspective', 'portion': (-0.08, 0.08), 'reference_size': 512, 'p': 0.25},
        {'name': 'rotate_hand', 'angle': (-5.0, 5.0), 'rotation_center': 'shoulder', 'p': 0.38},
        {'name': 'rotate_hand', 'angle': (-5.0, 5.0), 'rotation_center': 'elbow', 'p': 0.38},
        {'name': 'rotate_hand', 'angle': (-5.0, 5.0), 'rotation_center': 'wrist', 'p': 0.38},
        {'name': 'noise', 'std': 1.5, 'p': 0.5}
]

augmentations = {'none': [],
                 'light': light,
                 'medium': medium,
                 'heavy': heavy,
                 'individual-1': [medium[0]],
                 'individual-2': [medium[1]],
                 'individual-3': [medium[2]],
                 'individual-4': [medium[3]],
                 'individual-5': [medium[4]],
                 'individual-6': [medium[5]],
                 'individual-7': [medium[6]],
                 }

def get_augmentations(augmentation_type: str) -> list:
    return augmentations[augmentation_type]