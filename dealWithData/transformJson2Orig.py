import os
import json
import glob
from tqdm import tqdm


def convert_json_to_jsonl(input_directory, output_directory):
    """
    将输入目录中的json文件转换为标准jsonl格式，并保存到输出目录

    Args:
        input_directory: 包含原始json文件的目录
        output_directory: 保存转换后jsonl文件的目录
    """
    # 确保输出目录存在
    os.makedirs(output_directory, exist_ok=True)

    # 获取所有jsonl文件
    json_files = glob.glob(os.path.join(input_directory, "*.jsonl"))

    if not json_files:
        print(f"No .jsonl files found in {input_directory}")
        return

    print(f"Found {len(json_files)} files to process")

    for input_file in tqdm(json_files, desc="Converting files"):
        # 保持相同的文件名
        file_name = os.path.basename(input_file)
        output_file = os.path.join(output_directory, file_name)

        try:
            # 读取整个文件内容
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            # 处理文件
            try:
                # 尝试解析为单个JSON对象或JSON对象列表
                if content.startswith('[') and content.endswith(']'):
                    # 这是一个JSON数组
                    vessel_data_list = json.loads(content)

                    # 写入为JSONL格式
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        for vessel_data in vessel_data_list:
                            # 每行写入一个JSON对象
                            out_f.write(json.dumps(vessel_data) + '\n')
                else:
                    # 尝试按行解析，处理格式不规范的情况
                    lines = content.split('\n')
                    accumulated_json = ""
                    parsed_objects = []

                    for line in lines:
                        accumulated_json += line

                        # 尝试解析累积的JSON
                        try:
                            data = json.loads(accumulated_json)
                            parsed_objects.append(data)
                            accumulated_json = ""
                        except json.JSONDecodeError:
                            # 继续累积直到形成有效JSON
                            pass

                    # 写入解析出的对象
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        for obj in parsed_objects:
                            out_f.write(json.dumps(obj) + '\n')

            except json.JSONDecodeError as e:
                # 如果整体解析失败，使用更强健的方法
                print(f"Could not parse {file_name} as JSON: {e}")
                print("Attempting alternative parsing method...")

                # 尝试修复最常见的JSON格式问题
                try:
                    # 移除开头的"["和结尾的"]"，然后分割为单独的JSON对象
                    content = content.strip()
                    if content.startswith('['):
                        content = content[1:]
                    if content.endswith(']'):
                        content = content[:-1]

                    # 按"},{" 分割
                    parts = content.split("},")

                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        for i, part in enumerate(parts):
                            # 修复分割造成的格式问题
                            if i < len(parts) - 1:
                                part = part + "}"

                            # 确保每部分是有效的JSON
                            try:
                                vessel_data = json.loads(part)
                                out_f.write(json.dumps(vessel_data) + '\n')
                            except json.JSONDecodeError:
                                print(f"Skipping invalid JSON segment in {file_name}")

                except Exception as e2:
                    print(f"Alternative parsing failed for {file_name}: {e2}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print(f"Conversion complete. Processed files saved to {output_directory}")


def fix_specific_json_file(input_file, output_file):
    """
    修复特定的json文件并转换为标准jsonl格式

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    try:
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试解析为JSON
        try:
            data = json.loads(content)

            # 如果是列表，写入为JSONL
            if isinstance(data, list):
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    for item in data:
                        out_f.write(json.dumps(item) + '\n')
                print(f"Successfully converted {input_file} to JSONL format")
                return True
            else:
                # 单个对象
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(json.dumps(data) + '\n')
                print(f"Successfully wrote single JSON object to {output_file}")
                return True

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")

            # 手动修复常见问题
            if content.startswith('[') and ']' in content:
                # 可能是不完整的数组
                content = content.strip()
                if not content.endswith(']'):
                    content += ']'

                # 尝试修复缺失的逗号
                content = content.replace('}\n{', '},\n{').replace('} {', '}, {')

                try:
                    data = json.loads(content)
                    with open(output_file, 'w', encoding='utf-8') as out_f:
                        for item in data:
                            out_f.write(json.dumps(item) + '\n')
                    print(f"Fixed and converted {input_file}")
                    return True
                except json.JSONDecodeError as e2:
                    print(f"Could not fix JSON automatically: {e2}")

            # 最后的尝试：逐行读取并尝试解析
            try:
                with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
                    buffer = ""
                    brace_count = 0

                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        buffer += line

                        # 计算花括号的平衡
                        brace_count += line.count('{') - line.count('}')

                        # 当花括号平衡且buffer不为空，尝试解析
                        if brace_count == 0 and buffer:
                            try:
                                obj = json.loads(buffer)
                                out_f.write(json.dumps(obj) + '\n')
                                buffer = ""
                            except json.JSONDecodeError:
                                # 继续添加行直到形成有效JSON
                                pass

                if not buffer:  # 如果全部解析成功
                    print(f"Parsed {input_file} line by line successfully")
                    return True
                else:
                    print(f"Could not parse all content from {input_file}")
                    return False

            except Exception as e3:
                print(f"Line-by-line parsing failed: {e3}")
                return False

    except Exception as e:
        print(f"Error processing file: {e}")
        return False


def process_all_files():
    """主函数：处理所有文件"""
    input_dir = "../ship_trajectories"  # 原始json文件目录
    output_dir = "../ship_trajectories_orig"  # 修复后的jsonl文件目录

    convert_json_to_jsonl(input_dir, output_dir)

    # 如果需要处理特定文件
    # fix_specific_json_file("./ship_trajectories/0_4.jsonl", "./ship_trajectories_fixed/0_4.jsonl")


if __name__ == "__main__":
    process_all_files()