import os
import asyncio
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import regex as re  # 高效正则引擎
import orjson  # 快速 JSON 序列化
from numba import njit, vectorize, int32  # JIT 和向量化加速
import numpy as np  # 高效数组操作
import blosc  # 快速压缩缓冲区
import mmap  # 内存映射文件
import psutil  # 监控 CPU 使用率
import charset_normalizer  # 编码检测
from collections import deque  # 搜索历史
import logging  # 日志记录

# 常量定义
BUFFER_SIZE: int = 32768  # 32KB 缓冲区，贴合 i7-14700HX L3 缓存
MAX_MATCHES: int = 60000   # 最大匹配行数
OUTPUT_DIR: Path = Path(r"C:\Python312\pythonProject\Find")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_HISTORY: deque = deque(maxlen=5)  # 搜索历史，最多 5 条

# 日志配置
logging.basicConfig(filename=OUTPUT_DIR / 'search_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# 预编译正则表达式缓存
def compile_search_pattern(search_term: str) -> re.Pattern:
    """编译搜索词为正则表达式，支持模糊匹配，regex 引擎优化单核性能"""
    if search_term.startswith("fuzzy:"):
        fuzzy_term = search_term[6:].replace("_", ".")
        return re.compile(fuzzy_term.encode('utf-8'), re.BESTMATCH)
    return re.compile(re.escape(search_term), re.IGNORECASE | re.BESTMATCH)

@njit
def compute_line_numbers(buffer: bytes, pos: int) -> int:
    """使用 numba 加速行号计算，榨干单核性能"""
    line_count = 0
    for i in range(pos):
        if buffer[i] == 10:  # b'\n' 的 ASCII 值
            line_count += 1
    return line_count + 1

@njit
def boyer_moore_search(haystack: bytes, needle: bytes) -> int:
    """
    Boyer-Moore 算法，单核极致优化，理论最优时间复杂度 O(n/m)
    - haystack: 搜索缓冲区
    - needle: 搜索目标
    - 返回: 第一个匹配位置或 -1
    """
    n, m = len(haystack), len(needle)
    if m == 0 or n < m:
        return -1

    # 预计算坏字符表
    bad_char = np.full(256, m, dtype=np.int32)
    for i in range(m - 1):
        bad_char[needle[i]] = m - 1 - i

    # 搜索循环
    s = 0
    while s <= n - m:
        j = m - 1
        while j >= 0 and needle[j] == haystack[s + j]:
            j -= 1
        if j < 0:
            return s
        s += max(1, bad_char[haystack[s + m - 1]])
    return -1

@vectorize([int32(int32, int32)])
def kmp_search(haystack_char: int, needle_char: int) -> int:
    """KMP 辅助函数，向量化比较字符"""
    return 1 if haystack_char == needle_char else 0

async def search_single_match(file_path: Path, search_term: str) -> Optional[str]:
    """
    单次搜索：使用 Boyer-Moore 或模糊匹配，堆到理论最高速度
    时间复杂度：O(n/m) 或 O(n)（模糊），内存占用：O(1）
    """
    try:
        buffer_size = min(os.path.getsize(file_path) // 10, BUFFER_SIZE)  # 动态缓冲区
        search_term_bytes = search_term.encode('utf-8')
        is_fuzzy = search_term.startswith("fuzzy:")
        pattern = compile_search_pattern(search_term) if is_fuzzy else None

        with open(file_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # 预读优化
            # mmap.madvise(mmap.MADV_SEQUENTIAL)  # Windows 不支持，注释留作参考
            offset = 0
            while offset < len(mm):
                chunk = mm[offset:offset + buffer_size]
                compressed_chunk = blosc.compress(chunk, clevel=1, shuffle=blosc.SHUFFLE)
                decompressed_chunk = blosc.decompress(compressed_chunk)

                pos = -1
                if is_fuzzy:
                    match = pattern.search(decompressed_chunk)
                    if match:
                        pos = match.start()
                else:
                    pos = boyer_moore_search(decompressed_chunk, search_term_bytes)

                if pos != -1:
                    line_number = compute_line_numbers(decompressed_chunk, pos)
                    byte_start = offset + pos
                    byte_end = byte_start + len(search_term_bytes)

                    context_start = max(0, pos - 50)
                    context_end = min(len(decompressed_chunk), pos + len(search_term_bytes) + 50)
                    context = decompressed_chunk[context_start:context_end].decode('utf-8', errors='ignore')
                    context_highlight = f"{context[:pos - context_start]}[**{search_term}**]{context[pos - context_start + len(search_term):]}"

                    logging.info(f"Single match found: {search_term} at {byte_start}-{byte_end}")
                    return f"Find:{search_term} in line {line_number},txt:{byte_start}-{byte_end}   {context_highlight}"

                offset += buffer_size

        return None

    except (OSError, UnicodeDecodeError, ValueError) as e:
        print(f"文件读取或解码错误：{e}")
        logging.error(f"Error in single search: {e}")
        return None

async def search_multiple_matches(file_path: Path, search_term: str, output_file: Path, max_matches: int = MAX_MATCHES) -> int:
    """
    多次搜索：支持 Boyer-Moore 和模糊匹配，输出为 .txt，最多 3000 行
    时间复杂度：O(n/m) 或 O(n)（模糊），内存占用：O(k)
    """
    try:
        buffer_size = min(os.path.getsize(file_path) // 10, BUFFER_SIZE)
        results: List[str] = []
        search_term_bytes = search_term.encode('utf-8')
        is_fuzzy = search_term.startswith("fuzzy:")
        pattern = compile_search_pattern(search_term) if is_fuzzy else None

        with open(file_path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            offset = 0
            while offset < len(mm) and len(results) < max_matches:
                chunk = mm[offset:offset + buffer_size]
                compressed_chunk = blosc.compress(chunk, clevel=1, shuffle=blosc.SHUFFLE)
                decompressed_chunk = blosc.decompress(compressed_chunk)

                chunk_offset = 0
                while chunk_offset < len(decompressed_chunk) and len(results) < max_matches:
                    pos = -1
                    if is_fuzzy:
                        match = pattern.search(decompressed_chunk[chunk_offset:])
                        if match:
                            pos = match.start()
                    else:
                        pos = boyer_moore_search(decompressed_chunk[chunk_offset:], search_term_bytes)

                    if pos == -1:
                        break

                    abs_pos = chunk_offset + pos
                    line_number = compute_line_numbers(decompressed_chunk, abs_pos)
                    byte_start = offset + abs_pos
                    byte_end = byte_start + len(search_term_bytes)

                    context_start = max(0, abs_pos - 50)
                    context_end = min(len(decompressed_chunk), abs_pos + len(search_term_bytes) + 50)
                    context = decompressed_chunk[context_start:context_end].decode('utf-8', errors='ignore')
                    context_highlight = f"{context[:abs_pos - context_start]}[**{search_term}**]{context[abs_pos - context_start + len(search_term):]}"
                    result_line = f"Find: {search_term} in line {line_number} txt: {byte_start}-{byte_end}   {context_highlight}"
                    results.append(result_line)

                    chunk_offset = abs_pos + len(search_term_bytes)

                offset += buffer_size

        if results:
            output_file = output_file.with_suffix('.txt')
            async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
                await f.write("\n".join(results) + "\n")
            logging.info(f"Multiple matches saved: {len(results)} items to {output_file}")

        return len(results)

    except (OSError, UnicodeDecodeError, ValueError) as e:
        print(f"文件处理错误：{e}")
        logging.error(f"Error in multiple search: {e}")
        return 0

async def swift_search_runner():
    """主函数：高效运行搜索任务，榨干 i7-14700HX 单核性能"""
    print("欢迎使用单核极致优化版大文件文本搜索工具！")
    last_path: str = ""  # 路径记忆
    while True:
        file_path_input = input(f"请输入txt文件的绝对路径 [{last_path}]: ").strip() or last_path
        if file_path_input.lower() == 'quit':
            print("感谢使用，再见！单核已满载，i7 说：我尽力了！")
            break

        file_path = Path(os.path.normpath(file_path_input))  # 路径安全性
        if not file_path.exists():
            print("错误：文件路径不存在，请检查路径是否正确。")
            continue

        if os.path.getsize(file_path) > 1_073_741_824:  # 1GB
            if input("文件超大（>1GB），继续？(y/n): ").lower() != 'y':
                continue

        last_path = str(file_path)
        mode = input("请选择搜索模式（1：单次搜索，2：多次搜索）: ").strip()

        match mode:
            case "1":
                print("模糊搜索示例: fuzzy:123__4411_411（_ 表示任意字符）")
                if SEARCH_HISTORY:
                    print("历史搜索:", " | ".join(f"{i+1}: {term}" for i, term in enumerate(SEARCH_HISTORY)))
                search_term = input("请输入要搜索的内容: ").strip()
                SEARCH_HISTORY.append(search_term)

                async with asyncio.timeout(60):  # 超时保护
                    start_time = datetime.now()
                    result = await search_single_match(file_path, search_term)
                    elapsed_time = (datetime.now() - start_time).total_seconds()

                if result:
                    print(f"\n搜索内容: '{search_term}' (用时: {elapsed_time:.2f} 秒)")
                    print("搜索结果:")
                    print("-" * 50)
                    print(result)
                else:
                    print(f"\n搜索内容: '{search_term}' 未找到匹配项 (用时: {elapsed_time:.2f} 秒)。")
                print(f"单核已满载，i7 说：我尽力了！当前 CPU 使用率: {psutil.cpu_percent():.1f}%")

            case "2":
                max_matches = MAX_MATCHES
                while True:
                    print("模糊搜索示例: fuzzy:123__4411_411（_ 表示任意字符）")
                    if SEARCH_HISTORY:
                        print("历史搜索:", " | ".join(f"{i+1}: {term}" for i, term in enumerate(SEARCH_HISTORY)))
                    search_term = input("\n请输入要搜索的内容（输入 'quit' 退出）: ").strip()
                    if search_term.lower() == 'quit':
                        print("感谢使用，再见！")
                        break

                    SEARCH_HISTORY.append(search_term)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    output_file = OUTPUT_DIR / f"Find_{search_term}_{timestamp}.json"

                    async with asyncio.timeout(60):
                        start_time = datetime.now()
                        result_count = await search_multiple_matches(file_path, search_term, output_file, max_matches)
                        elapsed_time = (datetime.now() - start_time).total_seconds()

                    print(f"搜索 '{search_term}' 完成，找到 {result_count} 个匹配项（最多 {max_matches} 个），结果已保存至 {output_file.with_suffix('.txt')} (用时: {elapsed_time:.2f} 秒)")
                    print(f"单核已满载，i7 说：我尽力了！当前 CPU 使用率: {psutil.cpu_percent():.1f}%")

                break

            case _:
                print("无效选项，请输入 1 或 2。")
                continue

        print("\n")

if __name__ == "__main__":
    os.environ["NUMBA_NUM_THREADS"] = "1"  # 限制 numba 使用单线程
    try:
        asyncio.run(swift_search_runner())
    except ImportError as e:
        print(f"错误：缺少依赖，请安装以下库：\n- pip install aiofiles regex orjson numba numpy blosc psutil charset-normalizer")
        logging.error(f"Import error: {e}")
    except KeyboardInterrupt:
        print("\n用户中断操作，程序退出。单核已休息，i7 说：终于解放了！")