import os

folder_path = "yolo-class/Annotations/"

# フォルダ内のすべてのファイルを削除
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"ファイル {file_path} を削除中にエラーが発生しました: {e}")

folder_path = "yolo-class/JPEGImages/"

# フォルダ内のすべてのファイルを削除
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"ファイル {file_path} を削除中にエラーが発生しました: {e}")
