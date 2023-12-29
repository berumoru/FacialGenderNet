python3 ./tools/train.py \
--image_size 100 \
--classes "male,female" \
--zdata_path './dataset/UTKFace_train.zip' \
--result_dir "Result/try5" \
--num_epochs 100 \
--lr 0.0001 \
--batch_size 128 \
--save_checkpoint_per_epoch 10 \
--patience 10