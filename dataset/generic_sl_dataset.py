import os
from typing import Dict
import h5py
import json
import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, T5Tokenizer
import transformers
from collections import defaultdict


INPUT_TYPES = ["mae", "sign2vec", "dino", "pose"]


class SignFeatureDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        sign_data_args: dict,
        split: str = 'train',
        skip_frames=False,
        max_token_length=None,
        max_sequence_length=None,
        max_samples=None,
    ):
        """

        Args:
            tokenizer:
            sign_data_args:
            split:
            skip_frames:
            max_token_length:
            max_sequence_length:
        """
        self.skip_frames = skip_frames
        self.max_token_length = max_token_length
        self.max_sequence_length = max_sequence_length
        self.sign_data_args = sign_data_args
        self.tokenizer = tokenizer
        self.split = split
        self.max_samples = max_samples
        data_dir = sign_data_args['data_dir']

        assert self.split in ['train', 'dev', 'test'], 'split must be in ["train", "dev", "test"]'
        if self.split == "train":
            annotation_path = sign_data_args['annotation_path']['train']
        elif self.split == "dev":
            annotation_path = sign_data_args['annotation_path']['dev']
        elif self.split == "test":
            annotation_path = sign_data_args['annotation_path']['test']
        annotation_path = os.path.join(data_dir, annotation_path)
        with open(annotation_path, "r") as f:
            self.annotation = json.load(f)

        self.list_data = []  # [(video_id, clip_id), ...]
        self.h5_data = {}

        self.h5shard = defaultdict(lambda: defaultdict(dict))
        self.clip_order_to_int = {}
        self.clip_order_from_int = {}
        for video_id in self.annotation.keys():
            co = self.annotation[video_id]['clip_order']
            self.clip_order_from_int[video_id] =  dict(zip(range(len(co)),co))
            self.clip_order_to_int[video_id] =  dict(zip(co,range(len(co))))

        for video_id, clip_dict in self.annotation.items():
            for clip_name in clip_dict:
                if clip_name != "clip_order":
                    self.list_data.append((video_id, self.clip_order_to_int[video_id][clip_name]))
        for input_type in INPUT_TYPES:
            enable_feature = sign_data_args['visual_features'][input_type]['enable_input']
            if 'train' in sign_data_args['visual_features'][input_type]:
                vf_train_path = sign_data_args['visual_features'][input_type]['train']
            else:
                vf_train_path = None
            if 'dev' in sign_data_args['visual_features'][input_type]:
                vf_dev_path = sign_data_args['visual_features'][input_type]['dev']
            else:
                vf_dev_path = None
            if 'test' in sign_data_args['visual_features'][input_type]:
                vf_test_path = sign_data_args['visual_features'][input_type]['test']
            else:
                vf_test_path = None
            if enable_feature and vf_train_path is not None and vf_dev_path is not None:
                if self.split == "train":
                    h5_video_clip = self.read_multih5_json(data_dir, vf_train_path, input_type)
                    self.remove_missing_annotation(h5_video_clip)
                elif self.split == "dev":
                    h5_video_clip = self.read_multih5_json(data_dir, vf_dev_path, input_type)
                    self.remove_missing_annotation(h5_video_clip)
            elif self.split == "test" and enable_feature and vf_test_path is not None:
                h5_video_clip = self.read_multih5_json(data_dir, vf_test_path, input_type)
                self.remove_missing_annotation(h5_video_clip)
            else:
                self.h5_data[input_type] = None

        # Crop dataset to desired length
        if self.max_samples is not None:
            self.list_data = self.list_data[:self.max_samples]

    def __len__(self):
        return len(self.list_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        video_id, clip_id = self.list_data[i]
        clip_name = self.clip_order_from_int[video_id][clip_id]

        # Get the visual features
        visual_features = {}
        for input_type in INPUT_TYPES:
            if self.h5_data[input_type] is not None:
                shard = self.h5shard[self.split][input_type][video_id]
                vf = torch.tensor(np.array(self.h5_data[input_type][shard][video_id][clip_name]))

                visual_features[input_type] = vf
            else:
                visual_features[input_type] = None

        translation = self.annotation[video_id][clip_name]['translation']

        decoded = self.tokenizer(
            translation,
            max_length=self.max_token_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = decoded.input_ids

        # Skip frames for the keypoints
        if self.skip_frames:
            if type(self.skip_frames) == bool:
                for input_type in INPUT_TYPES:
                    if visual_features[input_type] is not None:
                        visual_features[input_type] = visual_features[input_type][::2]
            elif type(self.skip_frames) == int:
                for input_type in INPUT_TYPES:
                    if visual_features[input_type] is not None:
                        visual_features[input_type] = visual_features[input_type][::self.skip_frames]

        # Trim the keypoints to the max sequence length
        if self.max_sequence_length:
            for input_type in INPUT_TYPES:
                if visual_features[input_type] is not None:
                    visual_features[input_type] = visual_features[input_type][: self.max_sequence_length]
                    seq_len = len(visual_features[input_type])

        assert seq_len, "No modality provided or clip has no length!"
        attention_mask = torch.ones(seq_len)

        return {
            "sign_inputs": {'pose': visual_features['pose'],
                            'mae': visual_features['mae'],
                            'dino': visual_features['dino'],
                            'sign2vec': visual_features['sign2vec']},
            "sentence": translation,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def read_multih5_json(self, data_dir, json_filename, input_type):
        """Helper function for reading json specifications of multiple H5 files for visual features"""
        h5_video_clip = set()
        with open(os.path.join(data_dir, json_filename), 'r') as F:
            self.h5shard[self.split][input_type] = json.load(F)
            self.h5_data[input_type] = {}
            print(f"{input_type}: {self.split} data is loaded from: ")
            for k in set(self.h5shard[self.split][input_type].values()):
                h5file = os.path.join(data_dir,
                                      json_filename.replace('metadata_', '').replace('.json', ".%s.h5" % k))
                print("--" + h5file)  # ,k,json_filename,data_dir)
                self.h5_data[input_type][k] = h5py.File(h5file, 'r')

                for vi in self.h5_data[input_type][k].keys():
                    for ci in self.h5_data[input_type][k][vi].keys():
                        if vi in self.clip_order_to_int:
                            if ci in self.clip_order_to_int[vi]:
                                clip_id = self.clip_order_to_int[vi][ci]
                                h5_video_clip.add((vi, clip_id))
        return h5_video_clip

    def remove_missing_annotation(self, h5_video_clip):
        annotations_to_delete = set(self.list_data) - h5_video_clip
        for a in annotations_to_delete:
            self.list_data.remove(a)


if __name__ == '__main__':
    train_dataset = SignFeatureDataset(tokenizer= T5Tokenizer.from_pretrained('google-t5/t5-small'),
                                       sign_data_args={
                                'data_dir': '/media/zeleznyt/DATA/data/YTASL_small',
                                'annotation_path':
                                    {
                                    'train': 'YT.annotations.train.json',
                                    'dev': 'YT.annotations.dev.json',
                                    },
                                'visual_features':
                                    {
                                    'sign2vec':
                                        {
                                        'enable_input': False,
                                        'train': 'sign2vec/metadata_sign2vec.train.json',
                                        'dev': 'sign2vec/metadata_sign2vec.dev.json',
                                        },
                                    'mae':
                                        {
                                        'enable_input': False,
                                        'train': 'mae/metadata_mae.train.json',
                                        'dev': 'mae/metadata_mae.dev.json',
                                        },
                                    'dino':
                                        {
                                        'enable_input': False,
                                        'train': 'dino/metadata_dino.train.json',
                                        'dev': 'dino/metadata_dino.dev.json',
                                        },
                                    'pose':
                                        {
                                        'enable_input': True,
                                        'train': 'YouTubeASL.keypoints.train.json',
                                        'dev': 'YouTubeASL.keypoints.dev.json',
                                        }
                                    },
                                },
                                       split='train',
                                       )
    print(len(train_dataset))
    print(next(iter(train_dataset)))