# Vision-Based Word-Level Sign Language Recognition Using Deep Learning

## Motivation

For those with hearing impairments, sign language is an indispensable communication tool. Yet for many unfamiliar with its intricacies, it poses a comprehension barrier. This lack of understanding makes it even more important to have a reliable tool that can help translate sign language gestures, so that it can connect hearing-impaired community with those who don't know sign language.

## Methodolody

Our project targets the development of a deep learning-based tool to recognize American Sign Language (ASL) at the word level from videos. There are two main approaches regarding this task: holistic visual appearance-based and 2D human pose-based. Our focus for this project is the first approach, which emphasizes the modeling of both temporal and spatial aspects of video data to accurately interpret sign language.

### Base Model

<img src="base_model/base_model_architecture.jpeg" alt="Architecture of base model" height="300">

### I3D Model

<img width="857" alt="image" src="https://github.com/Allison67/sign-language-recognition/assets/96998345/cc1321b3-c5bf-4939-8ed2-03a8cf2594a7">

## Dataset

Download the dataset from [WLASL (World Level American Sign Language) Video | Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)

Our project utilizes the WLASL-100 dataset. This dataset is composed of 100 distinct ASL classes and is pre-divided into training and validation sets. The training set includes 748 videos and the validation set includes 165 videos.

## Environment setup

Install dependencies with `pip`:

```
pip install -r requirements.txt
```

## Overview of Script Files

- basemodel.py: The script for training the base model (including data pre-processing).
- basemodel_tuning.py: The script for hyperparameter tuning of the base model.
- i3d_training.py: The script for training the I3D model (including data pre-processing).
- i3d_tuning.py: The script for hyperparameter tuning of I3D model with various batch size and learning rate.

## Result

### Base Model

![Base model evaluation result](https://github.com/Allison67/sign-language-recognition/blob/main/base_model/basemodel_performance.png)
The training accuracy is approximately 100% while the validation accuracy is only 15.67%.

### I3D Model

![I3D evaluation result](https://github.com/Allison67/sign-language-recognition/blob/main/i3d/i3d_performance_without_tuning.png)
The validation accuracy for the I3D model before hyperparameter tuning is 51%.

### Hyperparameter Tuning for I3D Model

![hyperparameter tuning for i3d model](https://github.com/Allison67/sign-language-recognition/blob/main/i3d/hyperparameter_tuning_i3d.png)
We obtained the best configuration as batch size = 4 and learning rate = 0.001.

|              | lr=0.01 | lr=0.001 | lr=0.0001 |
| ------------ | ------- | -------- | --------- |
| batch_size=4 | 0.02    | 0.31     | 0.11      |
| batch_size=6 | 0.02    | 0.27     | 0.04      |
| batch_size=8 | 0.03    | 0.29     | 0.03      |

**Table:** Hyperparameter Tuning Result for I3D (Validation Accuracy).

### Post-tuning I3D Model

![performance of i3d after tuning](https://github.com/Allison67/sign-language-recognition/blob/main/i3d/I3D_after_tuning.png)
The validation accuracy for the I3D model after hyperparameter tuning is 61% now.

## Future Study Direction

- Expand dataset for richer sign language patterns.
- Explore different model architectures.
- Improve computational efficiency for practical use.

## References

- [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/pdf/1705.07750.pdf)
- [Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison](https://arxiv.org/abs/1910.11006)
- [WLASL Dataset](https://github.com/dxli94/WLASL)
