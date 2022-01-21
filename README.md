# Audio2Keypoint
PyTorch implementation of [Facial Keypoint Sequence Generation from Audio](https://arxiv.org/abs/2011.01114).

## Prerequisites
- python 3.9.6
- pytorch 1.9.0
- librosa 0.8.1
- matplotlib 3.4.2
- numpy 1.20.2
- pandas 1.2.5

The code might be compatible with older versions of these libraries. However, it has not been tested.

## Dataset
Download the Vox-KP dataset from [here](). The dataset is around 58 GB. Run the following code to extract log mel spectrogram of audio and to create temporal stack.
```
python prepro_feats.py --csv data/Vox-KP.csv
```
Each temporal stack containing the keypoints and spectrograms is saved as an `npz` file in `data/npz`.

## Training
To train the model, run the following code
```
python train.py --id audio2keypoint --csv data/infos.csv
```
Chekpoints are by default saved in `./checkpoints` directory. You can specify alternate path by passing `--checkpoint_path`. Only the best performing checkpoint on validation and the latest checkpoint are saved to minimize disk usage.

To resume training from the previous checkpoint, you can specify `--resume_training` to be `True`.

For a list of other options, see [opts.py](https://github.com/sidharth5n/Audio2Keypoint/blob/master/opts.py)

## Evaluation
To evalute the model on the `dev` split of Vox-KP dataset, run the following code.
```
python eval.py --id audio2keypoint --checkpoint_path './checkpoints' --split dev --output_path eval_results --csv data/infos.csv
```
The generated keypoints and comparative visualization of the generated and ground truth keypoints are saved in the directory provided in `--output_path`.

Visualization functions are taken from [speech2gesture](https://github.com/amirbar/speech2gesture).

To generate keypoints on an audio sample, run the following code
```
python eval.py --id audio2keypoint --checkpoint_path './checkpoints' --split audio --audio_path <path_to_file.wav> --image_path <path_to_driving_image.jpg> --output_path eval_results
```

For a list of other options, see [eval.py]()

## Reference
If you find this repository useful, please consider citing
```
@misc{manocha2020facial,
      title={Facial Keypoint Sequence Generation from Audio}, 
      author={Prateek Manocha and Prithwijit Guha},
      year={2020},
      eprint={2011.01114},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
