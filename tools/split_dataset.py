import zipfile
import random
import os

# Zipファイルの読み込み
zip_path = './dataset/UTKFace.zip'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()

    # Seedの設定とファイルリストのシャッフル
    random.seed(42)  # ここで任意のseed値を設定
    random.shuffle(file_list)

    # データ分割
    split_index = int(len(file_list) * 0.8)
    train_files = file_list[:split_index]
    test_files = file_list[split_index:]

    # 一時的なフォルダ作成
    temp_train_dir = './dataset/temp_train'
    temp_test_dir = './dataset/temp_test'
    os.makedirs(temp_train_dir, exist_ok=True)
    os.makedirs(temp_test_dir, exist_ok=True)

    # 一時フォルダにファイルの解凍
    for file in train_files:
        zip_ref.extract(file, temp_train_dir)
    for file in test_files:
        zip_ref.extract(file, temp_test_dir)

# 分割されたデータをZIP化
def zip_files(src_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(src_dir, '..')))

zip_files(temp_train_dir, './dataset/UTKFace_train.zip')
zip_files(temp_test_dir, './dataset/UTKFace_test.zip')

# 一時フォルダの削除
import shutil
shutil.rmtree(temp_train_dir)
shutil.rmtree(temp_test_dir)
print("Successfully!!")