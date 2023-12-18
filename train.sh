python3 ./tools/train.py \
--image_size = 128 \
--classes = ["male", "female"] \
--zdata_path = './dataset/UTKFace_test.zip' \
--result_dir = "Result/try1" \
--num_epochs =50 \
--lr = 0.001 \
--batch_size = 32 \
--save_checkpoint_per_epoch = 3 \
--patience = 5 #Early_stopping