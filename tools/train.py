import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import os, zipfile, io, re
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torch import nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import EarlyStopping, save_checkpoint, plot_and_save_training_results,forward_with_squeeze


def main(args):
    # ここで引数を使用した処理を行います
    print("========== config ==========")
    print(f"Image Size: {args.image_size}")
    args.classes = args.classes[0].split(',')
    print(f"Classes: {args.classes}")
    print(f"ZData Path: {args.zdata_path}")
    args.result_dir = f"{args.result_dir}/train"
    print(f"Result Directory: {args.result_dir}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Save Checkpoint Per Epoch: {args.save_checkpoint_per_epoch}")
    print(f"Patience for Early Stopping: {args.patience}")
    

    num_classes = len(args.classes)
    # GPUが利用可能かどうかをチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    X_train = X.to(torch.float32) / 255
    # one-hot変換
    y_train = F.one_hot(Y_tensor, num_classes=num_classes)

    # trainデータからvalidデータを分割
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        random_state = 42,
        stratify = y_train,
        test_size = 0.2
    )
    print(f"X_train:{X_train.shape}, y_train:{y_train.shape}, X_valid:{X_valid.shape}, y_valid:{y_valid.shape}") 

    # MobileNetV2モデルの読み込み（修正版）
    weights = MobileNet_V2_Weights.IMAGENET1K_V1  # または MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    # 最後の層を2クラス分類用にカスタマイズ
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    # 損失関数
    loss_function = nn.CrossEntropyLoss()
    
    # Fine Tuning
    for param in model.parameters(): # すべての層の活性化
        param.requires_grad = True
    # for name, child in model.named_children(): # TODO Fine Tuning なしのほうが精度いい
    #     if name == 'features':
    #         for idx, layer in enumerate(child):
    #             if idx < 13:
    #                 for param in layer.parameters():
    #                     param.requires_grad = False
    #             else:
    #                 break
    
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNetの平均と標準偏差に基づく正規化
        # 必要に応じて他の変換を追加
    ])
    # 検証データの変換（ランダムな変換を適用しない）
    valid_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Early stopping 設定
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    # Loss Functionの設定（BCEWithLogitsLossはシグモイドを内包している）
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)

    # データセットの変換
    X_train = X_train.permute(0, 3, 1, 2)
    X_train = torch.stack([train_transform(x) for x in X_train])
    X_valid = X_valid.permute(0, 3, 1, 2)
    X_valid = torch.stack([valid_transform(x) for x in X_valid])
    # データセットとDataLoaderの設定
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    
    # トレーニングループ
    try:
        print("========== train start ===========")
        model.to(device)
        for epoch in range(args.num_epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = torch.argmax(targets, dim=1) if targets.ndim > 1 else targets
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            # 学習データにおける損失と精度の計算
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total

            # バリデーションループ
            model.eval()
            valid_loss, valid_correct, valid_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    targets = torch.argmax(targets, dim=1) if targets.ndim > 1 else targets
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                    valid_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    valid_total += targets.size(0)
                    valid_correct += (predicted == targets).sum().item()

            valid_loss /= len(valid_loader)
            valid_accuracy = 100 * valid_correct / valid_total

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)
            
            # 特定のエポック間隔でチェックポイントを保存（最初のエポックは除く）
            if epoch > 0 and epoch % args.save_checkpoint_per_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=f"{args.result_dir}/weights/epoch{epoch}.pth")

            print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Train Acc: {train_accuracy}, Valid Acc: {valid_accuracy}")

            early_stopping(valid_loss)
            scheduler.step(valid_loss)
            if early_stopping.early_stop:
                break

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("========== train finish!! ==========")
        plot_and_save_training_results(train_losses, valid_losses, train_accuracies, valid_accuracies, f'{args.result_dir}/training_results.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--image_size", type=int, default=128, help="Size of the images")
    parser.add_argument("--classes", nargs='+', default="male,female", help="List of classes")
    parser.add_argument("--zdata_path", type=str, default='./dataset/UTKFace_train.zip', help="Path to the ZData")
    parser.add_argument("--result_dir", type=str, help="Directory to save results")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_checkpoint_per_epoch", type=int, default=3, help="Save checkpoint per number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    args = parser.parse_args()
    main(args)