python3 ./tools/test.py \
--image_size 100 \
--classes "male,female" \
--zdata_path './dataset/UTKFace_test.zip' \
--load_weight "Result/try5/train/weights/epoch10.pth" \
--batch_size 128 \
--result_dir "Result/try5"
