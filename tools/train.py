import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import os, zipfile, io, re, sys
import matplotlib.pyplot as plt
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
from tensorboardX import SummaryWriter
# os.chdir('C:/Users/skber/work/01_Defios/01_研修/FacialGenderNet')

# EarlyStopping クラスの実装
class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping")
        else:
            self.best_loss = val_loss
            self.counter = 0

# ModelCheckpointの実装
def save_checkpoint(state, filename):
    # ディレクトリのパスを取得
    dir_name = os.path.dirname(filename)
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # チェックポイントを保存
    torch.save(state, filename)


def main(args):
    # ここで引数を使用した処理を行います
    print(f"Image Size: {args.image_size}")
    print(f"Classes: {args.classes}")
    print(f"ZData Path: {args.zdata_path}")
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

    # ディレクトリの存在確認
    if os.path.exists(args.result_dir):
        print(f"Warning: Directory {args.result_dir} already exists. Exiting program.")
        sys.exit()
    # ZIP読み込み
    z_train = zipfile.ZipFile(args.zdata_path) # UTKFace_test.zip にしてる
    # 画像ファイルパスのみ取得
    imgfiles = [ x for x in z_train.namelist() if re.search(r".*jpg$", x)]

    X = []
    Y = []
    print("load dataset")
    for imgfile in tqdm(imgfiles):
        # ZIPから画像読み込み
        image = Image.open(io.BytesIO(z_train.read(imgfile)))
        # RGB変換
        image = image.convert('RGB')
        # リサイズ
        image = image.resize((args.image_size, args.image_size))
        # 画像から配列に変換
        data = np.asarray(image)
        file = os.path.basename(imgfile)
        file_split = [i for i in file.split('_')]
        X.append(data)
        Y.append(file_split[1])
    z_train.close()
    del z_train, imgfiles

    # tensor変換
    X_np = np.array(X)
    Y_np = np.array(Y, dtype=int)
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

    # すべての層をトレーニング可能にする
    for param in model.parameters():
        param.requires_grad = True
        
    # Data Augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # 必要に応じて他の変換を追加
    ])

    # Early stopping 設定
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # ReduceLROnPlateauの初期化
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)

    log_dir = f"{args.result_dir}/tf_log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # データセットの変換
    X_train = torch.stack([transform(x) for x in X_train])
    X_valid = torch.stack([transform(x) for x in X_valid])
    # データセットとDataLoaderの設定
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    
    # SummaryWriterのインスタンスを作成
    writer = SummaryWriter(log_dir=f"{args.result_dir}/tf_log")

    # トレーニングループ
    try:
        # モデルをデバイスに移動
        model.to(device)
        for epoch in range(args.num_epochs):  # num_epochsは設定するエポック数
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader: 
                inputs, targets = inputs.to(device), targets.to(device)
                if targets.dim() > 1:
                    targets = targets.argmax(dim=1)
                inputs = inputs.permute(0, 3, 1, 2)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 平均トレーニング損失
            train_loss /= len(train_loader)

            # 検証ループ
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if targets.dim() > 1:
                        targets = targets.argmax(dim=1)
                    inputs = inputs.permute(0, 3, 1, 2)
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                    valid_loss += loss.item()
                    # 正解率の計算（必要に応じて）

            # 平均検証損失
            valid_loss /= len(valid_loader)
            # Early StoppingとLearning Rate Schedulerの呼び出し
            early_stopping(valid_loss)
            scheduler.step(valid_loss)

            # Early Stoppingのチェック
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # チェックポイントの保存
            if epoch % args.save_checkpoint_per_epoch == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename= f"{args.result_dir}/weights/epoch{epoch}.pth.tar")

            # TensorBoardへのログ記録
            writer.add_scalar(f'{args.result_dir}/Loss/train', train_loss, epoch)
            writer.add_scalar(f'{args.result_dir}/Loss/val', valid_loss, epoch)
            # エポックごとの進捗の表示
            print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")
    except Exception as e:
        print(f"Error occurred: {e}")
        writer.close()
    finally:
        writer.close()
        print("train finish!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--image_size", type=int, default=128, help="Size of the images")
    parser.add_argument("--classes", nargs='+', default=["male", "female"], help="List of classes")
    parser.add_argument("--zdata_path", type=str, default='./dataset/UTKFace_test.zip', help="Path to the ZData")
    parser.add_argument("--result_dir", type=str, default="Result/try1", help="Directory to save results")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--save_checkpoint_per_epoch", type=int, default=3, help="Save checkpoint per number of epochs")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")

    args = parser.parse_args()
    main(args)