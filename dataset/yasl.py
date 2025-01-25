import os
import h5py
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from utils.normalization import normalize_yasl

POSE_LANDMARKS = [11, 12, 13, 14, 23, 24]

FACE_LANDMARKS = [
    0, 4, 13, 14, 17, 33, 37, 39, 46,
    52, 55, 61, 64, 81, 82, 93,
    133, 151, 152, 159, 172, 178, 181, 
    263, 269, 276, 282, 285, 291, 294,
    311, 323, 362, 386, 397,
    468, 473 
]

class YoutubeASLForPose(Dataset):

    def __init__(
        self,
        h5_file,
        transform = 'yasl',
        max_instances=None,
        is_normalized=False,
        zero_mean=False,
    ):
        self.transform = transform
        self.h5_file = h5py.File(h5_file, "r")
        self.max_instances = max_instances
        self.zero_mean = zero_mean

        self.is_normalized = is_normalized

        self.clip_idx2id = {}
        for idx, clip_id in enumerate(list(self.h5_file.keys())):
            self.clip_idx2id[idx] = clip_id

        self.idx2clip_id = {v: k for k, v in self.clip_idx2id.items()}

    def __len__(self):
        return len(list(self.h5_file.keys())) if self.max_instances is None else self.max_instances
    
    def get_item_by_clip_id(self, clip_id):
        video_id = clip_id.split('.')[0]
        keypoints = self.process_keypoints(video_id, clip_id)
        return keypoints

    def process_keypoints(self, video_id, clip_id):
        
        # Data difference between LUMI and Royal
        if video_id not in self.h5_file.keys():
            data = self.h5_file[clip_id]
        else:
            data = self.h5_file[video_id][clip_id]

        if self.is_normalized:
            data = np.array(data)
            keypoints = torch.tensor(data)
            # Replace NaN, Inf values with 0
            torch.nan_to_num_(keypoints, nan=0.0, posinf=0.0, neginf=0.0)
            return keypoints

        pose_landmarks = data["joints"]["pose_landmarks"][()]
        face_landmarks = data["joints"]["face_landmarks"][()]
        left_hand_landmarks = data["joints"]["left_hand_landmarks"][()]
        right_hand_landmarks = data["joints"]["right_hand_landmarks"][()]

        if self.transform:
            if self.transform == 'yasl':
                pose_landmarks = normalize_yasl(pose_landmarks)
                face_landmarks = normalize_yasl(face_landmarks)
                left_hand_landmarks = normalize_yasl(left_hand_landmarks)
                right_hand_landmarks = normalize_yasl(right_hand_landmarks)
            else:
                raise NotImplementedError(f'{self.transform} normalization is not implemented yet')

        # Select only the keypoints that are needed
        pose_landmarks = pose_landmarks[:, POSE_LANDMARKS, :]
        face_landmarks = face_landmarks[:, FACE_LANDMARKS, :]

        # Remove last 1 channel (visibility)
        pose_landmarks = pose_landmarks[:, :, :-1]
        face_landmarks = face_landmarks[:, :, :-1]
        left_hand_landmarks = left_hand_landmarks[:, :, :-1]
        right_hand_landmarks = right_hand_landmarks[:, :, :-1]

        # Convert keypoints to tensor
        pose_landmarks = torch.tensor(pose_landmarks, dtype=torch.float)
        face_landmarks = torch.tensor(face_landmarks, dtype=torch.float)
        left_hand_landmarks = torch.tensor(left_hand_landmarks, dtype=torch.float)
        right_hand_landmarks = torch.tensor(right_hand_landmarks, dtype=torch.float)

        # print(
        #     f"Pose: {pose_landmarks.shape}",
        #     f"Left Hand: {left_hand_landmarks.shape}",
        #     f"Right Hand: {right_hand_landmarks.shape}",
        #     f"Face: {face_landmarks.shape}",
        #     sep="\n"
        # )

        # Concatenate all keypoints
        keypoints = torch.cat(
            (pose_landmarks, left_hand_landmarks, right_hand_landmarks, face_landmarks),
            dim=1,
        )
        # Reduce the keypoints (T, N, C) -> (T, N*C)
        keypoints = keypoints.view(keypoints.size(0), -1) 
        # Check if keypoints are in the correct shape
        assert keypoints.shape[-1] == 255, "Key points are not in the correct shape"

        # Replace NaN values with 0
        torch.nan_to_num_(keypoints, nan=0.0)

        # Apply zero mean normalization to each keypoint dim
        if self.zero_mean:
            keypoints = (keypoints - keypoints.mean(dim=0)) / keypoints.std(dim=0)

        # Replace NaN, Inf values with 0
        torch.nan_to_num_(keypoints, nan=0.0, posinf=0.0, neginf=0.0)

        return keypoints

    def __getitem__(self, idx):
        
        clip_id = list(self.h5_file.keys())[idx]
        keypoints = self.process_keypoints(clip_id)

        return keypoints

class YoutubeASLForSign2Vec(Dataset):

    def __init__(
        self,
        h5_fpath,
        max_instances=None,
    ):
        self.h5_file = h5py.File(h5_fpath, "r")
        self.max_instances = max_instances

    def __len__(self):
        return len(list(self.h5_file.keys())) if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):
        
        data = self.h5_file[list(self.h5_file.keys())[idx]]

        sign2vec = data["features"][()]
        sentence = data["sentence"][()].decode("utf-8")

        return sign2vec, sentence

class YoutubeASLForSLT(YoutubeASLForPose, YoutubeASLForSign2Vec):

    def __init__(
        self,
        annotation_fpath,
        metadata_fpath,
        h5_fpath,
        mode="train",
        input_type="pose",
        skip_frames=True,
        transform="yasl",
        max_token_length=128,
        max_sequence_length=250,
        tokenizer="google-t5/t5-small",
        file_prefix="YouTubeASL",
        h5_prefix="YouTubeASL",
        max_instances=None,
        is_normalized=False,
        verbose=False,
    ):

        self.mode = mode
        self.verbose = verbose
        self.is_normalized = is_normalized

        self.max_instances = max_instances

        annotations = json.load(open(
            os.path.join(annotation_fpath, f'YT.annotations.{mode}.json')
        )) # YT.annotations.{mode}.json

        metadata = json.load(open(
            os.path.join(metadata_fpath, f'YouTubeASL.keypoints.{mode}.json')
        ))  # YT.keypoints.{mode}.json


        self.annotations = []
        for video_id in tqdm(annotations.keys()):
            clip_ids = annotations[video_id]['clip_order']
            for clip_id in clip_ids:

                if video_id not in metadata:
                    if verbose: print(f"Video id {video_id} not found in metadata")
                    continue

                h5_path = os.path.join(metadata_fpath, '.'.join([
                    h5_prefix, 'keypoints', mode, str(metadata[video_id]), 'h5'
                ]))

                if not os.path.exists(h5_path):
                    if verbose: print(f"File {h5_path} not found")
                    continue

                # Check clip_id in the h5 files
                h5_file = h5py.File(h5_path, "r")
                if not self.is_normalized:
                    if clip_id not in h5_file.keys():
                        if verbose: print(f"Clip id {clip_id} not found in {h5_path}")
                        continue

                if self.is_normalized:
                    if video_id not in h5_file.keys():
                        if verbose: print(f"video id {video_id} not found in {h5_path}")
                        continue
                    else:
                        if clip_id not in h5_file[video_id].keys():
                            if verbose: print(f"Clip id {clip_id} not found in {h5_path}")
                            continue

                self.annotations.append({
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "translation": annotations[video_id][clip_id]["translation"],
                    "h5_file": h5_path,
                })

        # Sort the annotations by the shard id
        self.annotations = sorted(self.annotations, key=lambda x: x["h5_file"])

        if max_instances is not None:
            self.annotations = self.annotations[:max_instances]
        
        print(f"Found {len(self.annotations)} annotations for {mode} set")

        self.shard_id = self.annotations[0]["h5_file"].split('.')[-2]
        self.h5_file_name = self.annotations[0]["h5_file"]

        self.input_type = input_type

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_token_length = max_token_length
        self.max_sequence_length = max_sequence_length
        self.skip_frames = skip_frames

        if self.input_type == "sign2vec" and skip_frames: raise ValueError("skip_frames should be False for `sign2vec` input")

        YoutubeASLForPose.__init__(self, self.h5_file_name, transform, max_instances, is_normalized)
        YoutubeASLForSign2Vec.__init__(self, self.h5_file_name, max_instances)

    def __len__(self):
        return len(self.annotations) if self.max_instances is None else self.max_instances

    def __getitem__(self, idx):
        # Get the keypoints and the sentence

        h5_file = self.annotations[idx]["h5_file"]
        file_idx = self.annotations[idx]["clip_id"]
        sentence = self.annotations[idx]["translation"]
        
        if self.input_type == "pose":
            # Reinitialize the dataset if the h5 file is different
            if self.h5_file_name != h5_file:
                if self.verbose: print(f"Reinitializing the dataset with {h5_file}")
                YoutubeASLForPose.__init__(self, h5_file, self.transform, self.max_instances, self.is_normalized)
            keypoints = self.get_item_by_clip_id(file_idx)

        elif self.input_type == "sign2vec":
            # Reinitialize the dataset if the h5 file is different
            if self.h5_file_name != h5_file:
                YoutubeASLForSign2Vec.__init__(self, h5_file, self.max_instances)
            keypoints = YoutubeASLForSign2Vec.get_item_by_clip_id(file_idx)

        self.h5_file_name = h5_file

        # Tokenize the sentence
        decoder_input_ids = self.tokenizer(
            sentence,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        # Shift the token ids using tokenizer._shift_tokens_right
        # decoder_input_ids = self.tokenizer._shift_right(decoder_input_ids)

        # Skip frames for the keypoints
        if self.skip_frames: keypoints = keypoints[::2]
        # Trim the keypoints to the max sequence length
        keypoints = keypoints[: self.max_sequence_length]
        attention_mask = torch.ones(len(keypoints))

        return {
            "sign_inputs": keypoints,
            "sentence": sentence,
            "labels": decoder_input_ids,
            "attention_mask": attention_mask,
        }

class YoutubeASLForSign2VecPretraining(YoutubeASLForPose):

    def __init__(
        self,
        h5_fpath,
        annotation_fpath,
        metadata_fpath,
        mode="train",
        transform="yasl",
        skip_frames=False,
        max_sequence_length=None,
        min_sequence_length=20,
        add_factor=1.0,
        zero_mean=False,
        max_instances=None,
        h5_prefix="YouTubeASL",
        input_type="pose",
        is_normalized=False,
    ):
        
        self.input_type = input_type

        self.mode = mode

        annotations = json.load(open(
            os.path.join(annotation_fpath, f'YT.annotations.{mode}.json')
        )) # YT.annotations.{mode}.json

        metadata = json.load(open(
            os.path.join(metadata_fpath, f'YouTubeASL.keypoints.{mode}.json')
        ))  # YT.keypoints.{mode}.json


        self.annotations = []
        for video_id in tqdm(annotations.keys()):
            clip_ids = annotations[video_id]['clip_order']
            for clip_id in clip_ids:

                if video_id not in metadata:
                    # print(f"Video id {video_id} not found in metadata")
                    continue

                h5_path = os.path.join(metadata_fpath, '.'.join([
                    h5_prefix, 'keypoints', mode, str(metadata[video_id]), 'h5'
                ]))

                if not os.path.exists(h5_path):
                    # print(f"File {h5_path} not found")
                    continue

                # Check clip_id in the h5 files
                h5_file = h5py.File(h5_path, "r")
                if clip_id not in h5_file.keys():
                    # print(f"Clip id {clip_id} not found in {h5_path}")
                    continue

                self.annotations.append({
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "translation": annotations[video_id][clip_id]["translation"],
                    "h5_file": h5_path,
                })

        # Sort the annotations by the shard id
        self.annotations = sorted(self.annotations, key=lambda x: x["h5_file"])

        if max_instances is not None:
            self.annotations = self.annotations[:max_instances]
        
        print(f"Found {len(self.annotations)} annotations for {mode} set")
        print(f"Using {self.annotations[0]['h5_file']} as the h5 file")

        self.h5_file_name = self.annotations[0]["h5_file"]

        self.transform = transform
        self.zero_mean = zero_mean

        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.skip_frames = skip_frames
        self.add_factor = add_factor
        self.is_normalized = is_normalized

        YoutubeASLForPose.__init__(self, self.h5_file_name, transform, zero_mean, is_normalized)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get the keypoints and the sentence

        h5_file = self.annotations[idx]["h5_file"]
        file_idx = self.annotations[idx]["clip_id"]
        sentence = self.annotations[idx]["translation"]
        
        if self.input_type == "pose":
            # Reinitialize the dataset if the h5 file is different
            if self.h5_file_name != h5_file:
                YoutubeASLForPose.__init__(self, h5_file, self.transform, self.max_instances, self.is_normalized)
            keypoints = self.get_item_by_clip_id(file_idx)
        else:
            raise ValueError("Only pose input is supported for pretraining")
            
        self.h5_file_name = h5_file

        if self.skip_frames: keypoints = keypoints[::2]
        if self.max_sequence_length: keypoints = keypoints[: self.max_sequence_length]

        keypoints = keypoints * self.add_factor

        # If less than min_sequence_length, pad with zeros

        if len(keypoints) < self.min_sequence_length:
            padding = torch.zeros(
                self.min_sequence_length - len(keypoints), keypoints.size(-1)
            )
            keypoints = torch.cat((keypoints, padding), dim=0)

        return {
            "input_values": keypoints,
        }
