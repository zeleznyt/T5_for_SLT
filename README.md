# T5_for_SLT
## Installation
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