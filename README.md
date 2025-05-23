# T5_for_SLT
## Publications

### Exploring Pose-based Sign Language Translation: Ablation Studies and Attention Insights
*Author1, Author2, ...*  
Published at *TBA*  
Code corresponding to this paper: [commit cb5fa58](https://github.com/zeleznyt/T5_for_SLT/tree/cb5fa58907b34365316f017ce6fe13d0116c829b)

**BibTeX citation**:
```bibtex
@inproceedings{TBA
}
```

### YouTube-ASL Clip Keypoint Dataset
We are releasing the YouTube-ASL keypoint dataset used in our research. It contains keypoints extracted from video clips that were publicly available at the time of collection and that passed our preprocessing and filtering pipeline. In total, the dataset includes 390,547 clips. It is publicly available at: http://hdl.handle.net/11234/1-5898.

**Dataset BibTeX citation**:
```
@misc{11234/1-5898,
  title     = {{YouTube}-{ASL} Clip Keypoint Dataset},
  author    = {Zelezny, Tomas and Hruz, Marek and Straka, Jaub and Gueuwou, Shester},
  url       = {http://hdl.handle.net/11234/1-5898},
  note      = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
  copyright = {Creative Commons - Attribution 4.0 International ({CC} {BY} 4.0)},
  year      = {2024}
}
```


## Installation
```
cd T5_for_SLT/
conda create -n t5slt python=3.12
conda activate t5slt
pip install -r requirements.txt
```
## Data
Our T5 dataloader expects the following data structure. The example is giver for YouTubeASL dataset.
This dataset consists of multiple videos. Each video consists of multiple clips. Each clip has its own annotation.
```
YT-ASL
|---YouTubeASL.annotation.train.json
|---YouTubeASL.annotation.dev.json
|---keypoints
|     |--- YouTubeASL.keypoints.train.json
|     |--- YouTubeASL.keypoints.train.0.h5
|     |--- YouTubeASL.keypoints.train.1.h5
|     |--- ...
|     |--- YouTubeASL.keypoints.dev.json
|     |--- YouTubeASL.keypoints.dev.0.h5
|     |--- YouTubeASL.keypoints.dev.1.h5
|     |--- ...
```
[//]: # (|---mae)

[//]: # (|     |--- yasl_mae_0.h5)

[//]: # (|     |--- ....)

[//]: # (|---dino)

[//]: # (|     |--- yasl_sign2vec_0.h5)

[//]: # (|     |--- ....)
Config file takes following inputs:
- Annotation file (```YouTubeASL.annotation.${split}.json```)
```
{
    ${video_id}: {
                "clip_order": [${clip_name}, ..., ],
                ${clip_name}: {
                                "translation": ....}
                ${clip_name}: ....,
                },
    ${video_id}: ...
}
```

- Metadata file (```YouTubeASL.${modality}.${split}.json```)
  - This file stores a dictionary with ```{video: shard_id}``` mapping for h5 shards.
  - Metadata file name **must be the same** as the shard names.

Each h5 shard has the following structure:
```
{
    ${video_id}: {
                    ${clip_name}: numpy.array (np.float16),
                    ${clip_name}: numpy.array, ...}
    ${video_id}:...
}
```

## Usage
```
# Setup correct path
cd T5_for_SLT/
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run training script
python train/run_finetuning.py --config_file configs/config.yaml

# Run predict script
python eval/predict.py --config_file configs/config.yaml
```
Use ```--verbose``` for extended output.

For more details see [scripts](scripts/) or [config readme](configs/README.md).
