import sys
import os
import time
from datetime import datetime
from PIL import ImageGrab, Image
import win32clipboard
import io
import subprocess
from PyQt5.QtWidgets import QApplication, QWidget, QRubberBand
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QScreen
from pynput import keyboard

# --- Configuration ---
SAVE_DIR_REGION = r"C:\Users\Xist_\Documents\oCam\捕获"
SAVE_DIR_FULL = r"C:\Users\Xist_\Documents\oCam\截屏"
REGION_TRIGGER_KEY = keyboard.Key.insert  # 区域截图触发键 (Insert)
FULL_SCREEN_TRIGGER_KEY = keyboard.Key.delete  # 全屏截图触发键 (Delete)
OPEN_LAST_DIR_KEY = keyboard.Key.end  # 打开上一张图片目录的触发键

# --- Ensure save directories exist ---
os.makedirs(SAVE_DIR_REGION, exist_ok=True)
os.makedirs(SAVE_DIR_FULL, exist_ok=True)

# --- Global variables ---
app_running = False
last_screenshot_dir = None


def log_message(message):
    """Prints log messages with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def copy_image_to_clipboard(image_pil):
    """Copies a PIL Image object to the clipboard."""
    try:
        output = io.BytesIO()
        image_pil.save(output, format="BMP")
        data = output.getvalue()

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        log_message("图片已成功复制到剪贴板。")
    except Exception as e:
        log_message(f"复制图片到剪贴板失败: {e}")


def open_folder(folder_path):
    """Opens the specified folder."""
    try:
        subprocess.Popen(f'explorer "{folder_path}"')
        log_message(f"已打开文件夹: {folder_path}")
    except Exception as e:
        log_message(f"打开文件夹失败: {e}")


def take_screenshot(region=None):
    """
    Performs the screenshot operation.
    If a region is provided, takes a regional screenshot; otherwise, a full-screen one.
    """
    global last_screenshot_dir

    timestamp_str = datetime.now().strftime("%m%d-%H%M")
    img = None
    filepath = None
    current_save_dir = None

    if region:
        # Regional screenshot
        x1, y1, x2, y2 = region
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        filename = f"捕获-{timestamp_str}.png"
        filepath = os.path.join(SAVE_DIR_REGION, filename)
        current_save_dir = SAVE_DIR_REGION
        log_message(f"正在捕获区域截图: {region}")
    else:
        # Full-screen screenshot
        img = ImageGrab.grab()
        filename = f"截屏-{timestamp_str}.png"
        filepath = os.path.join(SAVE_DIR_FULL, filename)
        current_save_dir = SAVE_DIR_FULL
        log_message("正在捕获全屏截图")

    if img:
        img.save(filepath)
        log_message(f"截图已保存到: {filepath}")
        copy_image_to_clipboard(img)
        last_screenshot_dir = current_save_dir
    else:
        log_message("未捕获到图像。")

    return filepath


class SelectionWindow(QWidget):
    """Transparent window for regional selection."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setWindowOpacity(0.3)
        self.setStyleSheet("background-color: black;")

        screen_geometry = QApplication.desktop().screenGeometry()
        self.setGeometry(screen_geometry)
        self.show()
        self.setCursor(Qt.CrossCursor)

        self.start_pos = None
        self.end_pos = None
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos()
            self.rubber_band.setGeometry(QRect(self.start_pos, self.start_pos))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            self.end_pos = event.globalPos()
            rect = QRect(self.start_pos, self.end_pos).normalized()
            self.rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event):
        global app_running
        if event.button() == Qt.LeftButton and self.start_pos:
            self.end_pos = event.globalPos()
            rect = QRect(self.start_pos, self.end_pos).normalized()

            self.hide()
            self.close()

            region = (rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height())
            take_screenshot(region)

            app_running = False
            QApplication.instance().quit()
            return


def start_region_screenshot():
    """Starts the regional screenshot tool."""
    global app_running
    if app_running:
        return

    app = QApplication(sys.argv)
    app_running = True
    ex = SelectionWindow()
    app.exec_()
    app_running = False


def on_press(key):
    """Keyboard press event listener."""
    global app_running, last_screenshot_dir

    if app_running:
        return

    try:
        if key == REGION_TRIGGER_KEY:
            log_message("检测到Insert键！正在启动区域截图...")
            start_region_screenshot()
        elif key == FULL_SCREEN_TRIGGER_KEY:
            log_message("检测到Delete键！正在执行全屏截图...")
            take_screenshot()
        elif key == OPEN_LAST_DIR_KEY:
            if last_screenshot_dir:
                log_message("检测到End键！正在打开上次截图的目录...")
                open_folder(last_screenshot_dir)
            else:
                log_message("没有找到上次截图的目录，请先进行一次截图。")
    except AttributeError:
        pass


def main():
    print("--- ScreenSnap 截图工具使用说明 ---")
    print(f"1. 按下并松开一次 {str(REGION_TRIGGER_KEY).replace('Key.', '')} 键 (Insert): 启动区域截屏工具。")
    print("   - 此时屏幕会变暗，用鼠标左键拖动以框选区域。")
    print("   - 松开鼠标左键后，选定区域的截图将自动保存并复制到剪贴板。")
    print(f"2. 按下 {str(FULL_SCREEN_TRIGGER_KEY).replace('Key.', '')} 键 (Delete): 立即截取全屏。")
    print("   - 全屏截图将自动保存并复制到剪贴板。")
    print(f"3. 按下 {str(OPEN_LAST_DIR_KEY).replace('Key.', '')} 键 (End): 打开上一次保存截图的目录。")
    print(f"\n截图将保存到以下目录:")
    print(f"  区域截屏: {SAVE_DIR_REGION}")
    print(f"  全屏截屏: {SAVE_DIR_FULL}")
    print("\n脚本正在运行，等待按键事件...")

    # Listen for all relevant keys on press
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()