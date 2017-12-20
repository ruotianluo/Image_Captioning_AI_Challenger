#! /bin/sh

#larger batch

id="dense_box_bn"$1
ckpt_path="log_"$id
if [ ! -d $ckpt_path ]; then
  mkdir $ckpt_path
fi
if [ ! -f $ckpt_path"/infos_"$id".pkl" ]; then
start_from=""
else
start_from="--start_from "$ckpt_path
fi

python train.py --id $id --caption_model denseatt --input_json data/chinese_talk.json --input_label_h5 data/chinese_talk_label.h5 --input_fc_dir data/chinese_bu_fc --input_att_dir data/chinese_bu_att --input_box_dir data/chinese_bu_box  --seq_per_img 5 --batch_size 50 --beam_size 1 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path $ckpt_path $start_from --save_checkpoint_every 3000 --language_eval 1 --val_images_use 10000 --max_epoch 37  --rnn_size 1300 --use_box 1 --use_bn 1 

if [ ! -d xe/$ckpt_path ]; then
cp -r $ckpt_path xe/
fi

python train.py --id $id --caption_model denseatt --input_json data/chinese_talk.json --input_label_h5 data/chinese_talk_label.h5 --input_fc_dir data/chinese_bu_fc --input_att_dir data/chinese_bu_att --input_box_dir data/chinese_bu_box --seq_per_img 5 --batch_size 50 --beam_size 1 --learning_rate 5e-5 --learning_rate_decay_start 0 --learning_rate_decay_every 55  --learning_rate_decay_rate 0.1  --scheduled_sampling_start 0 --checkpoint_path $ckpt_path --start_from $ckpt_path --save_checkpoint_every 3000 --language_eval 1 --val_images_use 10000 --self_critical_after 37 --rnn_size 1300 --use_bn 1 --use_box 1 
