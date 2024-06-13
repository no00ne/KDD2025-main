import zipfile
import os

def create_zip(zip_name, file_list, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建压缩包的完整路径
    zip_path = os.path.join(output_dir, zip_name)

    # 打包文件
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in file_list:
            zipf.write(file, os.path.basename(file))

    print(f'打包完成，压缩包路径: {zip_path}')