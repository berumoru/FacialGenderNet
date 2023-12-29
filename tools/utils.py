import os
import torch
import matplotlib.pyplot as plt

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

def plot_and_save_training_results(train_losses, valid_losses, train_accuracies, valid_accuracies, filename):
    plt.figure(figsize=(18, 6))

    # 正解率のプロット
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(valid_accuracies, label="Valid Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(color='gray', alpha=0.2)

    # 損失のプロット
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(valid_losses, label="Valid Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.grid(color='gray', alpha=0.2)

    # プロットを画像ファイルとして保存
    plt.savefig(filename)

    # プロットを表示
    plt.show()
    
def forward_with_squeeze(input,model):
    output = model(input)
    return output.squeeze()

def display_first_10_each(predictions_correct, predictions_incorrect, file_name):
    """correctとincorrectリストの最初の15個ずつを表示し、ファイルに保存する関数"""
    plt.figure(figsize=(16, 6))

    # 正しい予測の最初の10個を表示
    for i in range(min(10, len(predictions_correct))):
        plt.subplot(2, 10, i + 1)
        plt.axis("off")
        label = "male" if predictions_correct[i][1] == 0 else "female"
        plt.title(f'{label}\n{predictions_correct[i][2]:.2f}', color='black')
        image_data = unnormalize(predictions_correct[i][0]).permute(1, 2, 0).numpy()
        plt.imshow(image_data)

    # 間違った予測の最初の10個を表示
    for i in range(min(10, len(predictions_incorrect))):
        plt.subplot(2, 10, 10 + i + 1)
        plt.axis("off")
        label = "male" if predictions_incorrect[i][1] == 0 else "female"
        plt.title(f'{label}\n{predictions_incorrect[i][2]:.2f}', color='red')
        image_data = unnormalize(predictions_incorrect[i][0]).permute(1, 2, 0).numpy()
        plt.imshow(image_data)
    plt.savefig(file_name)  # 画像をファイルに保存
    
def display_predictions(predictions, file_name, is_correct=True):
    """予測された画像を表示し、ファイルに保存する関数"""
    # 並び替え：正解は確率の低い順、不正解は確率の高い順
    predictions.sort(key=lambda x: x[2], reverse=not is_correct)
    plt.figure(figsize=(16, 6))
    for i in range(min(30, len(predictions))):
        plt.subplot(3, 10, i + 1)
        plt.axis("off")
        label = "male" if predictions[i][1] == 0 else "female"
        title_color = 'red' if not is_correct else 'black'
        plt.title(f'{label}\n{predictions[i][2]:.2f}', color=title_color)
        # 正規化を逆に適用し、画像データの形状変換
        image_data = unnormalize(predictions[i][0]).permute(1, 2, 0).numpy()
        plt.imshow(image_data)
    plt.savefig(file_name)

# 正規化を逆に適用する関数
def unnormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image
