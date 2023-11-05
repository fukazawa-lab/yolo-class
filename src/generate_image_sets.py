import os
import random
import shutil
import argparse  # argparse ライブラリを追加
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

from PIL import Image

# コマンドライン引数の設定
parser = argparse.ArgumentParser()
parser.add_argument("--class_name", required=True, help="Class name for data processing")
args = parser.parse_args()

class_name = args.class_name  # コマンドライン引数から class_name を受け取る


# データ拡張の関数
def load_and_preprocess_image(image_path):
    target_size=(224, 224)
    # 画像を読み込み、指定したサイズにリサイズ
    image_path = class_name+"/output/train/"
    img = load_img(image_path, target_size=target_size)
    # 画像データをNumPy配列に変換
    img_array = img_to_array(img)
    # 画像データをモデルに合った形式で前処理
    preprocessed_image = preprocess_input(img_array)
    return preprocessed_image

def save_augmented_image(image, save_path):
    # NumPy配列からPIL Imageに変換
    augmented_img = Image.fromarray(image.astype('uint8'))
    # 画像を指定のパスに保存
    augmented_img.save(save_path)

# オリジナルのデータセットと新しいデータセットのフォルダを設定
source_folder = class_name+"/"  # 犬の画像がまとまっているフォルダ
output_folder = class_name+"/output/"  # 新しいデータセットのフォルダ

# 新しいデータセットのフォルダを作成
if not os.path.exists(class_name + "/output"):
    os.makedirs(class_name + "/output/train")
    os.makedirs(class_name + "/output/validation")
    os.makedirs(class_name + "/output/test")


# フォルダ内の画像ファイルをリストアップ
image_files = [f for f in os.listdir(source_folder) if f.endswith(".jpg")]

# ラベルの数を初期化
label = 0

# データセットを作成
for image_file in image_files:
    # 画像ファイルのパス
    source_path = source_folder+ image_file
    
    # 新しいファイル名を生成（CIFAR-10形式）
    new_filename = f"{class_name}_{label:04d}.jpg"
    
    # 新しいファイルの保存先パス
    target_path = class_name+"/output/train/"+new_filename
    
    # 画像ファイルのコピーとリネーム
    shutil.copy(source_path, target_path)
    
    # 次のラベルに進む
    label += 1

# データセットの分割（トレーニング、検証、テストセット）
images = os.listdir(class_name+"/output/train/")
train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

# # データ拡張を定義
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # トレーニングデータにデータ拡張を適用
# for image in train_images:
#     img_path = os.path.join(output_folder, image)
#     img = load_and_preprocess_image(img_path)
#     img = img.reshape((1,) + img.shape)  # バッチサイズ1に変更
#     i = 0
#     for batch in datagen.flow(img, batch_size=1):
#         save_path = os.path.join(output_folder, f"train_augmented_{i}_{image}")
#         save_augmented_image(batch[0], save_path)
#         i += 1
#         if i > 3:  # 4つの拡張画像を生成する例
#             break

# 検証データとテストデータを移動
for image in val_images:
  shutil.move(class_name+"/output/train/"+image, class_name+"/output/validation/"+image)

for image in test_images:
  shutil.move(class_name+"/output/train/"+image, class_name+"/output/test/"+ image)