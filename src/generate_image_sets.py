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
parser.add_argument("--class_id", required=True, help="Class id for data processing")
args = parser.parse_args()

class_name = args.class_name  # コマンドライン引数から class_name を受け取る
class_id = args.class_id  # コマンドライン引数から class_id を受け取る


# クラスごとにフォルダを作成
for class_id, class_name in enumerate(class_names):
#    os.system(f"python generate_image_sets.py --class_name={class_name} --class_id={class_id}")

  # オリジナルのデータセットと新しいデータセットのフォルダを設定
  source_folder = "yolo-class/image_source/"+class_name+"/"  # 犬の画像がまとまっているフォルダ
  annotations_folder = "yolo-class/Annotations/"  # アノテーションファイルを格納するフォルダ

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
      target_path = "yolo-class/JPEGImages/"+new_filename
      
      # 画像ファイルのコピーとリネーム
      shutil.copy(source_path, target_path)

      # 新しいファイルの保存先パス
      target_path = os.path.join(annotations_folder,  f"{class_name}_{label:04d}.txt")
      
      # アノテーションの内容
      annotation = f"{class_id} 0.01 0.01 0.98 0.98"  # YOLO形式のアノテーション
      
      # アノテーションファイルを書き込み
      with open(target_path, "w") as annotation_file:
          annotation_file.write(annotation)
      
      # 次のラベルに進む
      label += 1

  # データセットの分割（トレーニング、検証、テストセット）
  images = os.listdir("yolo-class/JPEGImages/")
  train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)


  # train_imagesのファイル名をtrain.txtに追記
  with open('yolo-class/src/data/custom/train.txt', 'w') as train_file:
      for image in train_images:
          train_file.write('/content/yolo-class/JPEGImages/' + image + '\n')

  # test_imagesのファイル名をvalid.txtに追記
  with open('yolo-class/src/data/custom/valid.txt', 'w') as valid_file:
      for image in test_images:
          valid_file.write('/content/yolo-class/JPEGImages/' + image + '\n')

  print("train.txtとvalid.txtにファイル名が追記されました。")
  sys.stdout.flush()

