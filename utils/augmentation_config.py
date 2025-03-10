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
individual_1 = [
        {'name': 'rotate', 'angle': (-3.0, 3.0), 'p': 0.5}
]
individual_2 = [
        {'name': 'shear', 'angle_x': (-3.0, 3.0), 'angle_y': (-3.0, 3.0), 'p': 0.38}
]
individual_3 = [
        {'name': 'perspective', 'portion': (-0.08, 0.08), 'reference_size': 512, 'p': 0.25}
]
individual_4 = [
        {'name': 'rotate_hand', 'angle': (-5.0, 5.0), 'rotation_center': 'shoulder', 'p': 0.38}
]
individual_5 = [
        {'name': 'rotate_hand', 'angle': (-5.0, 5.0), 'rotation_center': 'elbow', 'p': 0.38}
]
individual_6 = [
        {'name': 'rotate_hand', 'angle': (-5.0, 5.0), 'rotation_center': 'wrist', 'p': 0.38}
]
individual_7 = [
        {'name': 'noise', 'std': 1.5, 'p': 0.5}
]

augmentations = {'none': [],
                 'light': light,
                 'medium': medium,
                 'heavy': heavy,
                 'individual_1': individual_1,
                 'individual_2': individual_2,
                 'individual_3': individual_3,
                 'individual_4': individual_4,
                 'individual_5': individual_5,
                 'individual_6': individual_6,
                 'individual_7': individual_7,
                 }

def get_augmentations(augmentation_type: str) -> list:
    return augmentations[augmentation_type]