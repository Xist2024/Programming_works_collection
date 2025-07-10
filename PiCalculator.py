# 优化版本 v1.7: 异步计算与 I/O，动态 FPS，流式搜索
import os
import time
import sys
import asyncio
from datetime import datetime
from typing import Optional, Tuple
import mpmath  # pip install mpmath
import pygame  # pip install pygame
from functools import lru_cache
from pathlib import Path

# 初始化 Pygame 和窗口
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 300, 200
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pi Calculator v1.7")
clock = pygame.time.Clock()

# 开关按钮参数
SWITCH_WIDTH, SWITCH_HEIGHT = 60, 30
SWITCH_X, SWITCH_Y = (WINDOW_WIDTH - SWITCH_WIDTH) // 2, (WINDOW_HEIGHT - SWITCH_HEIGHT) // 2
switch_on = False
circle_x = SWITCH_X + 5
target_x = circle_x

# 设置保存 pi 值的目录
PI_DIR = Path(r"C:\Python312\pythonProject\pi")
try:
    PI_DIR.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(f"错误: 创建目录 {PI_DIR} 失败: {e}")
    pygame.quit()
    sys.exit(1)

# 初始化全局变量
pi_file: Optional[Path] = None
enter_count = 0
computing = False
precision = 1000  # 初始精度 1000 位
search_mode = False
search_query = ""
compute_start_time = 0.0

# 日志输出函数
def print_log(precision: int, time_taken: float, enter_count: int) -> None:
    print(f"精度: {precision:>6} 位 | 用时: {time_taken:>7.3f} 秒 | Enter 计数: {enter_count}/3")

# 异步计算 pi
@lru_cache(maxsize=2)  # 缓存最近两次结果
async def compute_pi(precision: int) -> Tuple[str, float]:
    mpmath.mp.dps = precision
    start_time = time.time()
    pi_value = str(mpmath.pi)
    time_taken = time.time() - start_time
    await asyncio.sleep(0)  # 让出控制权
    return pi_value, time_taken

# 异步保存 pi 值到文件
async def save_pi_to_file(filename: Path, pi_value: str) -> bool:
    try:
        async with aiofiles.open(filename, 'w', encoding='utf-8') as f:  # 异步写入
            for chunk in (pi_value[i:i+8192] for i in range(0, len(pi_value), 8192)):
                await f.write(chunk)
        return True
    except OSError as e:
        print(f"错误: 保存文件 {filename} 失败: {e}")
        return False

# 搜索 pi 值中的子串（流式处理）
def search_pi_in_file(filename: Path, query: str) -> Tuple[int, int]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            buffer = ""
            offset = 0
            decimal_found = False
            for chunk in iter(lambda: f.read(8192), ''):
                buffer += chunk
                if not decimal_found and '.' in buffer:
                    decimal_start = buffer.index('.') + 1
                    buffer = buffer[decimal_start:]
                    decimal_found = True
                pos = buffer.find(query)
                if pos != -1:
                    return pos + 1, pos + len(query)
                offset += len(chunk)
                buffer = buffer[-len(query):]  # 固定缓冲区大小
            return -1, -1
    except OSError as e:
        print(f"错误: 读取文件 {filename} 失败: {e}")
        return -1, -1

# 主循环（异步运行）
async def main_loop():
    global switch_on, circle_x, target_x, computing, precision, pi_file, enter_count, search_mode, search_query
    print("状态: 打开开关以开始计算 pi。按 Enter 键三次以退出程序。")
    print("=" * 70)
    running = True

    while running:
        # 处理 Pygame 事件
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    running = False
                case pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if (SWITCH_X <= mouse_x <= SWITCH_X + SWITCH_WIDTH and
                            SWITCH_Y <= mouse_y <= SWITCH_Y + SWITCH_HEIGHT):
                        switch_on = not switch_on
                        target_x = SWITCH_X + SWITCH_WIDTH - 35 if switch_on else SWITCH_X + 5
                        if switch_on:
                            computing = True
                            precision = 1000
                            start_time = datetime.now()
                            filename = start_time.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
                            pi_file = PI_DIR / filename
                            compute_start_time = time.time()
                            print(f"状态: 开始新计算，保存至 {filename}")
                            print("=" * 70)
                        else:
                            computing = False
                            search_mode = True
                            search_query = ""
                            print("状态: 开关关闭，进入搜索模式")
                case pygame.KEYDOWN if not switch_on and search_mode:
                    match event.key:
                        case pygame.K_RETURN if search_query:
                            if pi_file:
                                start_pos, end_pos = search_pi_in_file(pi_file, search_query)
                                if start_pos != -1:
                                    print(f"它处在第 {start_pos} 位至第 {end_pos} 位之间")
                                else:
                                    print("未找到")
                            search_query = ""
                        case pygame.K_BACKSPACE:
                            search_query = search_query[:-1]
                        case _ if event.unicode.isprintable():
                            search_query += event.unicode
                    print(f"请查询: {search_query}", end="\r", flush=True)

        # 开关动画
        if circle_x != target_x:
            circle_x += (target_x - circle_x) * 0.2
            if abs(circle_x - target_x) < 0.1:
                circle_x = target_x

        # 绘制界面
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, (0, 255, 0) if switch_on else (0, 0, 0),
                         (SWITCH_X, SWITCH_Y, SWITCH_WIDTH, SWITCH_HEIGHT), border_radius=15)
        pygame.draw.circle(screen, (255, 255, 255), (int(circle_x), SWITCH_Y + SWITCH_HEIGHT // 2), 12)
        pygame.display.flip()
        clock.tick(10 if computing else 30)  # 动态 FPS

        # 计算和保存 pi
        if switch_on and computing:
            pi_value, time_taken = await compute_pi(precision)
            print_log(precision, time_taken, enter_count)
            if await save_pi_to_file(pi_file, pi_value):
                precision +=50000000
            else:
                print("警告: 保存失败，检查磁盘空间或权限")
                computing = False

        # 退出逻辑
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RETURN]:
            enter_count += 1
            print(f"通知: 已按 Enter，总计: {enter_count}/3")
            if enter_count >= 3:
                print("=" * 70)
                print("状态: 检测到三次 Enter，程序退出。")
                print("=" * 70)
                running = False
            await asyncio.sleep(0.2)  # 异步防抖

        await asyncio.sleep(0.01)  # 让出控制权

# 运行主循环
if __name__ == "__main__":
    try:
        import aiofiles  # pip install aiofiles
        asyncio.run(main_loop())
    except ImportError:
        print("错误: 需要安装 aiofiles (pip install aiofiles)")
    finally:
        pygame.quit()
        sys.exit(0)