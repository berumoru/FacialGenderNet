import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import os, zipfile, io, re
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from utils import display_first_10_each, display_predictions

def main(args):
    # ここで引数を使用した処理を行います
    print("========== config ==========")
    print(f"Image Size: {args.image_size}")
    args.classes = args.classes[0].split(',')
    print(f"Classes: {args.classes}")
    print(f"ZData Path: {args.zdata_path}")
    print(f"Load weight file: {args.load_weight}")
    print(f"Batch Size: {args.batch_size}")
    args.result_dir = f"{args.result_dir}/test"
    print(f"Load Train Dir Directory: {args.result_dir}")

    num_classes = len(args.classes)
    # GPUが利用可能かどうかをチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 保存先のディレクトリが存在するか確認し、存在しなければ作成
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    # ZIP読み込み
    z_train = zipfile.ZipFile(args.zdata_path) # UTKFace_test.zip にしてる
    # 画像ファイルパスのみ取得
    imgfiles = [ x for x in z_train.namelist() if re.search(r".*jpg$", x)]

    X = []
    Y = []
    print("========== load dataset ===========")
    for imgfile in tqdm(imgfiles):
        image = Image.open(io.BytesIO(z_train.read(imgfile))) # ZIPから画像読み込み
        image = image.convert('RGB') # RGB変換
        image = image.resize((args.image_size, args.image_size)) # リサイズ
        # 画像から配列に変換
        data = np.asarray(image)
        file = os.path.basename(imgfile)
        file_split = [i for i in file.split('_')]
        X.append(data)
        Y.append(file_split[1])
    z_train.close()
    del z_train, imgfiles

    # tensor変換
    # 条件に合致する Y の要素のインデックスを取得
    valid_indices = [i for i, y in enumerate(Y) if y in ['0', '1']]
    Y_np = np.array([int(Y[i]) for i in valid_indices], dtype=int) # 条件に合致する Y の要素のみを選択
    X_np = np.array([X[i] for i in valid_indices]) # 同じインデックスの X の要素を選択
    X = torch.tensor(X_np)
    Y_tensor = torch.tensor(Y_np, dtype=torch.long)

    # データ型の変換＆正規化
    X_test = X.to(torch.float32) / 255
    # one-hot変換
    y_test = F.one_hot(Y_tensor, num_classes=num_classes)

    print(f"X_test:{X_test.shape}, y_train:{y_test.shape}") 

    # モデルのアーキテクチャを定義
    saved_model = torch.load(args.load_weight) # 保存された重みを読み込み
    model_state_dict = saved_model['state_dict']
    # MobileNetV2モデルのインスタンスを作成
    model = models.mobilenet_v2(weights=None) #事前学習された重みはロードしない
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    # state_dictをモデルにロード
    model.load_state_dict(model_state_dict)

    # 検証データの変換（ランダムな変換を適用しない）
    test_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    X_test = X_test.permute(0, 3, 1, 2)
    X_test = torch.stack([test_transform(x) for x in X_test])
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 推論プロセス
    try:
        print("========== start evaluation !! ===========")
        model = model.to('cuda')
        model.eval()

        test_correct, test_total = 0, 0
        correct, incorrect = [], []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                targets = torch.argmax(targets, dim=1) if targets.ndim > 1 else targets
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

                # 正誤判定とリストへの追加
                for i in range(targets.size(0)):
                    prob = torch.nn.functional.softmax(outputs, dim=1)[i, predicted[i]].item()
                    if predicted[i] == targets[i]:
                        correct.append((inputs[i].cpu(), predicted[i], prob))
                    else:
                        incorrect.append((inputs[i].cpu(), predicted[i], prob))
        print("========== finish evaluation !! ===========")
        # correctとincorrectの最初の15個ずつを表示し、保存
        display_first_10_each(correct, incorrect, f'{args.result_dir}/predictions_first_each10.png')
        print(f"save: {args.result_dir}/predictions_first_each10.png")
        # 正解画像の表示と保存
        display_predictions(correct, f'{args.result_dir}/correct_predictions.png', is_correct=True)
        print(f"save: {args.result_dir}/correct_predictions.png")
        # 不正解画像の表示と保存
        display_predictions(incorrect, f'{args.result_dir}/incorrect_predictions.png', is_correct=False)
        print(f"save: {args.result_dir}/incorrect_predictions.png")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("========== Done ===========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--image_size", type=int, default=128, help="Size of the images")
    parser.add_argument("--classes", nargs='+', default="male,female", help="List of classes")
    parser.add_argument("--zdata_path", type=str, default='./dataset/UTKFace_train.zip', help="Path to the ZData")
    parser.add_argument("--load_weight", type=str, help="File to load weight")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--result_dir", type=str, help="Directory to save results")
    

    args = parser.parse_args()
    main(args)