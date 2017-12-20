# Image Captioning in Chinese (trained on AI Challenger)

This provides the code to reproduce my result on AI Challenger Captioning contest (#3 on test b).

This is based on my [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) repository and [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch). (They all share a lot of the same git history)

## Requirements
Python 2.7 
PyTorch 0.2 (along with torchvision)
tensorboard-pytorch
jieba
hashlib

## Pretrained models (not supported)

## Train your own network on AI Challenger

### Download ai_challenger dataset and preprocessing

First, download the ai_challenger images from [link](https://challenger.ai/competition/caption/subject). We need both training and validationd data. We decompress the data into a same folder, say `data/ai_challenger`, the structure would look like:

```
├── data
│   ├── ai_challenger
│   │   ├── caption_train_annotations_20170902.json
│   │   ├── caption_train_images_20170902
│   │   │   ├── ...
│   │   ├── caption_validataion_annotations_20170910.json
│   │   ├── caption_validation_images_20170910
│   │   │   ├── ...
│   ├── ...

```

Once we have the images and the annotations, we can now invoke the `prepro_*.py` script, which will read all of this in and create a dataset (two feature folders, a hdf5 label file and a json file).

```bash
$ python scripts/prepro_split_tokenize.py --input_json ./data/ai_challenger/caption_train_annotations_20170902.json ./data/ai_challenger/caption_validation_annotations_20170910.json --output_json ./data/data_chinese.json --num_val 10000 --num_test 10000
$ python scripts/prepro_labels.py --input_json data/data_chinese.json --output_json data/chinese_talk.json --output_h5 data/chinese_talk --max_length 20 --word_count_threshold 20
$ python scripts/prepro_reference_json.py --input_json ./data/ai_challenger/caption_train_annotations_20170902.json ./data/ai_challenger/caption_validation_annotations_20170910.json --output_json ./data/eval_reference.json
$ python scripts/prepro_ngrams.py --input_json data/data_chinese.json --dict_json data/chinese_talk.json --output_pkl data/chinese-train --split train

```

`prepro_split_tokenize` will conbine both training and validation data, and randomly the dataset into train, val and test. It will also tokenize the captions using jiebe.

`prepro_labels.py` will map all words that occur <= 20 times to a special `卍` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/chinese_talk.json` and discretized caption data are dumped into `data/chinese_talk_label.h5`.

`prepro_reference_json.py` will prepare the json file for caption evaluation.

`prepro_ngrams.py` will prepare the file for self critical training.

(Check the prepro scripts for more options, like other resnet models or other attention sizes.)

### Prepare the features

We use bottom-up features to get the best results. However, if the code should also support using resnet101 features.

- Using resnet101

```
$ python scripts/prepro_feats.py --input_json data/data_chinese.json --output_dir data/chinese_talk --images_root data/ai_challenger --att_size 7
```

This extracts the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/chinese_talk_fc` and `data/chinese_talk_att`, and resulting files are about 100GB.

- Using bottom-up-features

Here is the pre-extracted feature for downloading [link](https://drive.google.com/open?id=1kap1IqsNXSiKqkrGcyDMyso0h9SMnyvq).

Code for extracting the features is [here](https://github.com/ruotianluo/bottom-up-attention-ai-challenger)

### Download the evaluation code

Clone from [link](https://github.com/AIChallenger/AI_Challenger) and [link](https://github.com/ruotianluo/cider)

### Start training

```bash
mkdir xe
$ bash run_train.sh
```

### Evaluate on test split

```bash
$ python eval.py --dump_images 0 --num_images -1 --split test  --model log_dense_box_bn/model-best.pth --language_eval 1 --beam_size 5 --temperature 1.0 --sample_max 1  --infos_path log_dense_box_bn/infos_dense_box_bn-best.pkl
```

To run ensemble:

```
python eval_ensemble.py --dump_images 0 --language_eval 1 --batch_size 5 --num_images -1 --split test  --ids dense_box_bn dense_box_bn1 --beam_size 5 --temperature 1.0 --sample_max 1
```

## Acknowledgements

Thanks the original [neuraltalk2](https://github.com/karpathy/neuraltalk2) and awesome PyTorch team.