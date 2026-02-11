import re
import json
import os

def process_srt_content(content):
    """处理单个SRT文件的文本逻辑"""
    # 替换敏感词 [cite: 2, 7]
    processed_text = re.sub(r'\bnigga\b', 'NEGA', content, flags=re.IGNORECASE)
    
    # 提取对话行，跳过序号和时间戳
    lines = processed_text.split('\n')
    dialogue_segments = []
    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        dialogue_segments.append(line)
    
    # 构造多轮对话 [cite: 1, 3, 4, 5, 6]
    formatted_dialogue = []
    for i, text in enumerate(dialogue_segments):
        role = "User: " if i % 2 == 0 else "Assistant: "
        formatted_dialogue.append(f"{role}{text}")
    
    # 用 \n\n 分隔轮次
    return "\n\n".join(formatted_dialogue)

def batch_convert_to_jsonl(input_dir, output_file):
    """批量转换目录下所有 srt 文件"""
    results = []
    
    # 获取所有 .srt 后缀的文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.srt')]
    
    if not files:
        print("未找到任何 .srt 文件")
        return

    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            final_text = process_srt_content(content)
            # 每一行是一个完整的 JSON 对象，包含该文件的所有对话
            results.append({"text": final_text})
            print(f"成功处理: {file_name}")
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    # 写入 JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\n--- 批量处理完成 ---")
    print(f"共转换 {len(results)} 个文件 -> {output_file}")

if __name__ == "__main__":
    # '.' 代表当前脚本所在的文件夹
    batch_convert_to_jsonl('.', 'batch_output.jsonl')
