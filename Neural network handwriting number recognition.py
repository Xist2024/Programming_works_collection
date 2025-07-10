# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import mixed_precision # Mixed precision typically not beneficial for CPU
from sklearn.model_selection import train_test_split
import matplotlib

# Use a backend compatible with tkinter BEFORE importing pyplot
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk # Import themed widgets
from tkinter import font as tkFont # For better font handling
from PIL import Image, ImageDraw, ImageOps
import time
import os
import sys
import traceback # Import for better error reporting

# --- Windows Specific Imports ---
if sys.platform == 'win32':
    try:
        import win32gui
        import win32com.client
        import win32api
        print("成功导入 pywin32 模块。")
    except ImportError:
        print("警告：未找到 'pywin32' 库。发送到记事本的功能将不可用。")
        print("请在Windows上运行 'pip install pywin32' 来安装它。")
        win32gui = None
        win32com = None
        win32api = None
else:
    print("信息：当前操作系统不是Windows，发送到记事本的功能将被禁用。")
    win32gui = None
    win32com = None
    win32api = None
# --- End Windows Specific Imports ---


# --- Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 200
CONTINUATION_EPOCHS = 70
BATCH_SIZE = 512
VERBOSE_LEVEL = 1
SAVE_DIR = "mnist_recognition_results_cpu_optimized"
CANVAS_SIZE = 336
AUGMENT_FACTOR = 2

# --- Style Configuration for GUI ---
STYLE_CONFIG = {
    "BG_COLOR": "#ECECEC", "FRAME_BG_COLOR": "#FFFFFF", "TEXT_COLOR": "#333333",
    "PRIMARY_COLOR": "#0078D4", "PRIMARY_ACTIVE": "#005A9E", "PRIMARY_TEXT": "#FFFFFF",
    "SECONDARY_COLOR": "#D83B01", "SECONDARY_ACTIVE": "#A22C00", "SECONDARY_TEXT": "#FFFFFF",
    "CANVAS_BG": "#000000", "CANVAS_BORDER": "#BBBBBB", "LABEL_FG": "#111111",
    "PRED_FG": "#005A9E", "FONT_FAMILY": "Segoe UI", "FONT_FALLBACK": "Arial",
    "FONT_SIZE_NORMAL": 10, "FONT_SIZE_LARGE": 12, "PAD_X_MAIN": 20, "PAD_Y_MAIN": 20,
    "PAD_X_WIDGET": 10, "PAD_Y_WIDGET": 5, "PAD_X_INTERNAL": 8, "PAD_Y_INTERNAL": 4,
    "BORDER_WIDTH": 1,
}

# --- Function Definitions ---

def get_font(size='normal', weight='normal'):
    """Helper function to get a Tkinter font object."""
    font_family = STYLE_CONFIG["FONT_FAMILY"]
    try:
        tk.font.nametofont("TkDefaultFont").config(family=font_family)
    except tk.TclError:
        font_family = STYLE_CONFIG["FONT_FALLBACK"]
    size_val = STYLE_CONFIG["FONT_SIZE_LARGE"] if size == 'large' else STYLE_CONFIG["FONT_SIZE_NORMAL"]
    weight_val = tkFont.BOLD if weight == 'bold' else tkFont.NORMAL
    return tkFont.Font(family=font_family, size=size_val, weight=weight_val)


def generate_mnist_data(augment_factor=AUGMENT_FACTOR):
    """Load and prepare the MNIST dataset with moderate data augmentation."""
    # --- Command Line Output (保持不变) ---
    print("\n--- 加载并增强MNIST数据集 ---")
    try:
        (X_train_orig, y_train_orig), (X_test, y_test) = keras.datasets.mnist.load_data()
    except Exception as e:
        print(f"错误：无法加载 MNIST 数据集。请检查网络连接或数据集缓存。错误信息: {e}")
        sys.exit(1) # Exit if data cannot be loaded

    # --- 保存原始数据 ---
    X_train_original_flat = X_train_orig.reshape(-1, 28 * 28).astype('float32') / 255.0
    y_train_original_cat = keras.utils.to_categorical(y_train_orig, num_classes=10)

    X_train_combined = [X_train_original_flat]
    y_train_combined = [y_train_original_cat]

    # --- CPU优化: 适度增强 ---
    if augment_factor > 0:
        print(f"进行数据增强 (因子: {augment_factor})...")
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
            zoom_range=0.1, fill_mode='nearest'
        )
        X_train_orig_reshaped = (X_train_orig.astype('float32') / 255.0).reshape(-1, 28, 28, 1)
        start_aug_time = time.time()
        for i in range(augment_factor):
            print(f"  生成第 {i+1}/{augment_factor} 批增强数据...")
            try:
                gen = datagen.flow(X_train_orig_reshaped, y_train_orig, batch_size=len(X_train_orig), shuffle=False)
                X_batch, y_batch = next(gen)
                X_train_aug_flat = X_batch.reshape(-1, 28 * 28)
                y_train_aug_cat = keras.utils.to_categorical(y_batch, num_classes=10)
                X_train_combined.append(X_train_aug_flat)
                y_train_combined.append(y_train_aug_cat)
            except Exception as e:
                print(f"数据增强过程中发生错误 (批次 {i+1}): {e}")
                traceback.print_exc()
                # Decide whether to continue or exit
                # For now, we continue without this batch
                print("跳过此批增强数据。")
                continue # Skip to the next augmentation factor

        print(f"增强数据生成完毕，耗时: {time.time() - start_aug_time:.2f} 秒")

        # 合并所有成功生成的数据
        try:
            X_train_final = np.concatenate(X_train_combined, axis=0)
            y_train_final = np.concatenate(y_train_combined, axis=0)
            print(f"合并后用于分割的总样本数: {X_train_final.shape[0]} 样本")
        except ValueError as e:
            print(f"错误: 无法合并训练数据。可能是由于增强过程中出错导致数据为空。错误: {e}")
            # Fallback to original data if concatenation fails
            X_train_final = X_train_original_flat
            y_train_final = y_train_original_cat
            print("将仅使用原始数据进行训练。")

    else:
        print("未进行数据增强 (augment_factor <= 0)")
        X_train_final = X_train_original_flat
        y_train_final = y_train_original_cat


    # Reshape and One-Hot Encode Test Data
    X_test = X_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    y_test = keras.utils.to_categorical(y_test, num_classes=10)

    # Split final data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_final, y_train_final, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )

    # --- Command Line Output (保持不变) ---
    print(f"最终训练集大小: {X_train.shape[0]} 样本")
    print(f"最终验证集大小: {X_val.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape):
    """Build the handwritten digit recognition model (三层隐藏层)."""
    # --- Command Line Output (保持不变) ---
    print("\n--- 构建手写数字识别模型 ---")
    try:
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,), name="input_layer"),
            keras.layers.Dense(256, activation='relu', name="hidden_layer_1"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu', name="hidden_layer_2"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu', name="hidden_layer_3"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(10, activation='softmax', name="output_layer", dtype='float32')
        ], name="mnist_recognition_model_cpu_optimized")
        model.summary()
        return model
    except Exception as e:
        print(f"构建模型时出错: {e}")
        traceback.print_exc()
        sys.exit(1)


def compile_and_train(model, X_train, y_train, X_val, y_val, save_dir, continue_training=False):
    """Compile and train the model, optimized for CPU utilization."""
    # --- Command Line Output (保持不变) ---
    print("\n--- 编译和训练模型 ---")
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    except Exception as e:
        print(f"编译模型时出错: {e}")
        traceback.print_exc()
        return None, model # Return None for history if compile fails

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    checkpoint_path = os.path.join(save_dir, "best_model_epoch_cpu.keras")
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)

    # --- CPU优化: 使用 tf.data 并添加 .cache() ---
    try:
        print("创建 tf.data 数据管道 (包含缓存)...")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).cache().shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        print("数据管道创建完毕。")
    except Exception as e:
        print(f"创建 tf.data 数据管道时出错: {e}")
        print("可能是内存不足以缓存数据。尝试减小 BATCH_SIZE 或移除 .cache()")
        traceback.print_exc()
        return None, model # Return None for history

    # 设置训练轮数
    epochs_to_run = CONTINUATION_EPOCHS if continue_training else EPOCHS
    if continue_training:
        print(f"继续训练... (最多 {epochs_to_run} 轮, Batch Size: {BATCH_SIZE})")
    else:
        print(f"开始训练... (最多 {epochs_to_run} 轮, Batch Size: {BATCH_SIZE})")

    history = None # Initialize history
    start_time = time.time()
    try:
        history = model.fit(
            train_dataset,
            epochs=epochs_to_run,
            validation_data=val_dataset,
            verbose=VERBOSE_LEVEL,
            callbacks=[early_stopping, lr_scheduler, model_checkpoint]
        )
        training_time = time.time() - start_time
        print(f"总训练时间: {training_time:.2f} 秒 (注意: 由于缓存，第一个epoch可能较慢)")
    except Exception as e:
        print(f"模型训练过程中出错: {e}")
        traceback.print_exc()
        # history might be partially filled, or None
        # Model state might be inconsistent, saving might fail or save bad state

    # --- 保存模型 ---
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_save_path = os.path.join(save_dir, "final_model_cpu.keras")
        model.save(model_save_path)
        print(f"最终模型已保存至: {model_save_path}")
        print(f"训练过程中最佳验证准确率模型已保存至: {checkpoint_path} (如果生成)")
    except Exception as e:
        print(f"保存最终模型时出错: {e}")
        traceback.print_exc()

    # --- 加载回最佳模型 ---
    best_model_loaded = False
    if os.path.exists(checkpoint_path):
        try:
            print(f"加载回训练过程中验证准确率最高的模型: {checkpoint_path}")
            # It's generally safer to load into a new variable if memory allows,
            # or ensure the session is clear if overwriting 'model' directly.
            # For simplicity here, we overwrite.
            model = keras.models.load_model(checkpoint_path)
            best_model_loaded = True
        except Exception as e:
            print(f"加载最佳模型失败: {e}，将使用最终（可能未完成训练）的模型。")
            traceback.print_exc()

    if not best_model_loaded:
         print("将使用训练结束时（或出错时）的最终模型进行后续操作。")

    return history, model # Return potentially partial history and the best/final model


def load_saved_model(save_dir, filename="best_model_epoch_cpu.keras"):
    """Load the saved complete model"""
    model_path = os.path.join(save_dir, filename)
    final_model_path = os.path.join(save_dir, "final_model_cpu.keras")
    loaded_model = None
    print(f"尝试加载模型: {model_path}")
    if os.path.exists(model_path):
        try:
            loaded_model = keras.models.load_model(model_path)
            print(f"\n--- 已成功加载模型从 {model_path} ---")
            return loaded_model
        except Exception as e:
            print(f"加载模型 {model_path} 失败: {e}")
            traceback.print_exc()
    else:
        print(f"模型文件 {model_path} 不存在。")

    # 如果最佳模型加载失败或不存在，尝试加载最终模型
    print(f"尝试加载最终模型: {final_model_path}")
    if loaded_model is None and os.path.exists(final_model_path):
         try:
             loaded_model = keras.models.load_model(final_model_path)
             print(f"\n--- 已成功加载最终模型从 {final_model_path} ---")
             return loaded_model
         except Exception as e_final:
             print(f"加载最终模型 {final_model_path} 失败: {e_final}")
             traceback.print_exc()
    else:
         if loaded_model is None: # Only print if final model also doesn't exist
              print(f"最终模型文件 {final_model_path} 也不存在。")


    if loaded_model is None:
        print("未找到可加载的模型文件。")
        print("将重建新模型并重新训练")
    return loaded_model # Returns None if loading failed


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on the test set using tf.data"""
    if model is None:
        print("评估错误：模型对象为 None。")
        return
    # --- Command Line Output (保持不变) ---
    print("\n--- 评估模型性能 ---")
    try:
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        loss, accuracy = model.evaluate(test_dataset, verbose=0)
        # --- Command Line Output (保持不变) ---
        print(f"测试集损失: {loss:.4f}")
        print(f"测试集准确率: {accuracy:.4f}")
    except Exception as e:
        print(f"模型评估过程中出错: {e}")
        traceback.print_exc()


# --- DrawingApp Class ---
# (代码与上一个版本相同，确认无语法错误)
class DrawingApp:
    """Tkinter GUI for drawing digits and getting predictions (Enhanced Style & Layout)"""
    def __init__(self, root, model):
        # Check if model is valid before proceeding
        if model is None:
            print("错误：DrawingApp 无法初始化，模型为 None。")
            # Optionally show an error message in the GUI window itself
            tk.Label(root, text="Error: Model not loaded or trained successfully.", fg="red", font=("Segoe UI", 14)).pack(padx=20, pady=20)
            root.title("Error")
            # Keep root alive briefly to show the message? Or just exit?
            # For now, the calling code handles the exit if model is None after main logic.
            return # Stop initialization

        self.root = root
        self.model = model
        self.root.title("Handwritten Digit Recognition")
        self.root.configure(bg=STYLE_CONFIG["BG_COLOR"])
        self.root.resizable(True, True)
        self.root.minsize(650, 450)

        self.font_normal = get_font('normal')
        self.font_bold = get_font('normal', 'bold')
        self.font_large_bold = get_font('large', 'bold')

        self.main_frame = tk.Frame(root, bg=STYLE_CONFIG["BG_COLOR"])
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=STYLE_CONFIG["PAD_X_MAIN"], pady=STYLE_CONFIG["PAD_Y_MAIN"])

        self.main_frame.grid_columnconfigure(0, weight=2)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.canvas_labelframe = ttk.LabelFrame(self.main_frame, text=" Drawing Area ", labelanchor="nw", padding=(15, 10))
        self.canvas_labelframe.grid(row=0, column=0, sticky="nsew", padx=(0, STYLE_CONFIG["PAD_X_WIDGET"]))

        self.canvas_labelframe.grid_rowconfigure(0, weight=0)
        self.canvas_labelframe.grid_rowconfigure(1, weight=1)
        self.canvas_labelframe.grid_rowconfigure(2, weight=0)
        self.canvas_labelframe.grid_columnconfigure(0, weight=1)

        self.canvas_container = tk.Frame(self.canvas_labelframe, bg=STYLE_CONFIG["FRAME_BG_COLOR"], bd=STYLE_CONFIG["BORDER_WIDTH"], relief="sunken")
        self.canvas_container.grid(row=1, column=0, sticky='nsew', pady=(5, STYLE_CONFIG["PAD_Y_WIDGET"]))

        self.canvas = tk.Canvas(self.canvas_container, width=CANVAS_SIZE, height=CANVAS_SIZE, bg=STYLE_CONFIG["CANVAS_BG"], highlightthickness=0)
        self.canvas.pack(padx=5, pady=5)

        self.button_frame = tk.Frame(self.canvas_labelframe)
        self.button_frame.grid(row=2, column=0, pady=(STYLE_CONFIG["PAD_Y_WIDGET"], 0))

        style = ttk.Style()
        style.configure("Predict.TButton", font=self.font_bold, background=STYLE_CONFIG["PRIMARY_COLOR"], foreground=STYLE_CONFIG["PRIMARY_TEXT"], padding=(STYLE_CONFIG["PAD_X_INTERNAL"], STYLE_CONFIG["PAD_Y_INTERNAL"]))
        style.map("Predict.TButton", background=[('active', STYLE_CONFIG["PRIMARY_ACTIVE"])])
        self.predict_button = ttk.Button(self.button_frame, text="Predict", command=self.predict, style="Predict.TButton", width=10)
        self.predict_button.pack(side=tk.LEFT, padx=(0, STYLE_CONFIG["PAD_X_WIDGET"] // 2))

        style.configure("Clear.TButton", font=self.font_bold, background=STYLE_CONFIG["SECONDARY_COLOR"], foreground=STYLE_CONFIG["SECONDARY_TEXT"], padding=(STYLE_CONFIG["PAD_X_INTERNAL"], STYLE_CONFIG["PAD_Y_INTERNAL"]))
        style.map("Clear.TButton", background=[('active', STYLE_CONFIG["SECONDARY_ACTIVE"])])
        self.clear_button = ttk.Button(self.button_frame, text="Clear", command=self.clear, style="Clear.TButton", width=10)
        self.clear_button.pack(side=tk.LEFT, padx=(STYLE_CONFIG["PAD_X_WIDGET"] // 2, 0))

        self.result_labelframe = ttk.LabelFrame(self.main_frame, text=" Predictions ", labelanchor="nw", padding=(15, 10))
        self.result_labelframe.grid(row=0, column=1, sticky="nsew", padx=(STYLE_CONFIG["PAD_X_WIDGET"], 0))
        self.result_labelframe.grid_columnconfigure(0, weight=0)
        self.result_labelframe.grid_columnconfigure(1, weight=1)

        self.prob_labels = []
        for i in range(10):
            digit_lbl = tk.Label(self.result_labelframe, text=f"{i}:", font=self.font_bold, fg=STYLE_CONFIG["LABEL_FG"])
            digit_lbl.grid(row=i, column=0, sticky="w", padx=(0, 5), pady=STYLE_CONFIG["PAD_Y_WIDGET"] // 2)
            prob_lbl = tk.Label(self.result_labelframe, text=" 0.00%", font=self.font_normal, fg=STYLE_CONFIG["PRED_FG"], anchor='w')
            prob_lbl.grid(row=i, column=1, sticky="ew", padx=(5, 0), pady=STYLE_CONFIG["PAD_Y_WIDGET"] // 2)
            self.prob_labels.append(prob_lbl)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.line_width = int(CANVAS_SIZE / 12)

    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y
        radius = self.line_width // 4
        if radius < 1: radius = 1 # Ensure radius is at least 1
        try:
            self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill='white', outline='white')
            self.draw_obj.ellipse([event.x - radius, event.y - radius, event.x + radius, event.y + radius], fill=255, outline=255)
        except Exception as e:
            print(f"Error during start_draw: {e}")

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            try:
                self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                        fill='white', width=self.line_width, capstyle=tk.ROUND, smooth=tk.TRUE)
                self.draw_obj.line([self.last_x, self.last_y, event.x, event.y], fill=255, width=self.line_width)
            except Exception as e:
                 print(f"Error during draw: {e}")
        self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            radius = self.line_width // 4
            if radius < 1: radius = 1
            try:
                self.canvas.create_oval(self.last_x - radius, self.last_y - radius, self.last_x + radius, self.last_y + radius, fill='white', outline='white')
                self.draw_obj.ellipse([self.last_x - radius, self.last_y - radius, self.last_x + radius, self.last_y + radius], fill=255, outline=255)
            except Exception as e:
                 print(f"Error during stop_draw: {e}")
        self.last_x, self.last_y = None, None

    def clear(self):
        try:
            self.canvas.delete("all")
            self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
            self.draw_obj = ImageDraw.Draw(self.image)
            for lbl in self.prob_labels:
                lbl.config(text=" 0.00%", font=self.font_normal) # Reset font too
        except Exception as e:
            print(f"Error during clear: {e}")

    def predict(self):
        """Predict the digit and potentially output to Notepad."""
        if self.model is None:
             print("预测错误：模型未加载。")
             return
        bbox = self.image.getbbox()
        if not bbox:
             print("画布为空或内容过少，请先绘制数字。")
             return

        try:
            img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28 * 28)
            prediction = self.model.predict(img_array, verbose=0)[0]
            predicted_digit = np.argmax(prediction)
            for i, prob in enumerate(prediction):
                font_to_use = self.font_bold if i == predicted_digit else self.font_normal
                self.prob_labels[i].config(text=f" {prob * 100:.2f}%", font=font_to_use)
            self.output_to_notepad(predicted_digit)
        except Exception as e:
            print(f"预测过程中发生错误: {e}")
            traceback.print_exc()
            # Consider showing a user-friendly error in the GUI

    def output_to_notepad(self, digit_to_output):
        """Finds an open Notepad window and sends the predicted digit as a keystroke."""
        if win32gui is None or win32com is None: return
        hwnd = None
        possible_titles = ["无标题 - 记事本", "Untitled - Notepad"]
        for title in possible_titles:
            hwnd = win32gui.FindWindow(None, title)
            if hwnd: break
        if not hwnd: hwnd = win32gui.FindWindow("Notepad", None)

        if hwnd:
            print(f"找到记事本窗口 (句柄: {hwnd})")
            try:
                shell = win32com.client.Dispatch("WScript.Shell")
                if shell.AppActivate(hwnd):
                    time.sleep(0.15)
                    shell.SendKeys(str(digit_to_output))
                    print(f"已发送数字 '{digit_to_output}' 到记事本。")
                else:
                    print("AppActivate 失败，尝试 SetForegroundWindow...")
                    try:
                         win32gui.SetForegroundWindow(hwnd)
                         time.sleep(0.15)
                         shell.SendKeys(str(digit_to_output))
                         print(f"通过 SetForegroundWindow 发送数字 '{digit_to_output}' 到记事本。")
                    except win32api.error as e_fg:
                         print(f"SetForegroundWindow 也失败: {e_fg}")
                         print("请尝试手动点击记事本窗口再进行预测。")
            except Exception as e:
                print(f"发送按键到记事本时发生未知错误: {e}")
                traceback.print_exc()
        else:
            print("未检测到打开的记事本窗口。")
# --- End DrawingApp Class ---


# --- plot_training_history Function ---
# (代码与上一个版本相同, 确认无语法错误)
def plot_training_history(history):
    """Plot training history curves (Accuracy and Loss) - English Labels"""
    print("\n--- 绘制训练历史曲线 ---")
    if not history or not hasattr(history, 'history') or not history.history:
        print("错误：无法绘制训练历史，history 对象无效或为空。")
        return
    required_keys = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    if not all(key in history.history for key in required_keys):
        print(f"错误：history 对象缺少必要的键。需要: {required_keys}。 找到: {list(history.history.keys())}。")
        return

    try:
        plt.figure(figsize=(12, 5))
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except (OSError, ImportError):
            print("未找到 'seaborn-v0_8-darkgrid' 风格，使用默认风格")
            plt.style.use('default')
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='#0078D4')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#E81123')
        plt.title('Model Accuracy', fontsize=12)
        plt.xlabel('Epoch', fontsize=10); plt.ylabel('Accuracy', fontsize=10)
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss', color='#0078D4')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='#E81123')
        plt.title('Model Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=10); plt.ylabel('Loss', fontsize=10)
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"绘制图表时出错: {e}")
        traceback.print_exc()

def main():
    """Main function - Optimized for CPU utilization with Enhanced GUI"""
    overall_start_time = time.time()

    print("--- 开始执行手写数字识别脚本 ---")
    print(f"TensorFlow 版本: {tf.__version__}")
    print(f"Keras 版本: {keras.__version__}")
    print(f"运行平台: {sys.platform}")
    print(f"CPU优化设置: AUGMENT_FACTOR={AUGMENT_FACTOR}, EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, 使用tf.data.cache()")
    print("\n--- 默认使用 float32 进行CPU训练 ---")

    # --- Data Loading and Preparation ---
    data_prep_start_time = time.time()
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = generate_mnist_data()
    except Exception as e:
        print(f"数据准备阶段发生严重错误: {e}")
        traceback.print_exc()
        sys.exit(1)
    print(f"数据加载和准备耗时: {time.time() - data_prep_start_time:.2f} 秒")

    model = None
    history = None

    # --- Model Loading/Training Logic ---
    best_model_path = os.path.join(SAVE_DIR, "best_model_epoch_cpu.keras")
    final_model_path = os.path.join(SAVE_DIR, "final_model_cpu.keras")
    model_exists = os.path.exists(best_model_path) or os.path.exists(final_model_path)

    if model_exists:
        choice = input(f"检测到已保存的模型在 {SAVE_DIR}，加载后继续训练(c)还是直接测试(t)？(c/t): ").strip().lower()
        if choice == 't':
             model = load_saved_model(SAVE_DIR)
             if model is None:
                 print("加载失败，将重新构建和训练模型。")
                 model = build_model(X_train.shape[1])
                 if model: # Check if build succeeded
                      history, model = compile_and_train(model, X_train, y_train, X_val, y_val, SAVE_DIR, continue_training=False)
        elif choice == 'c':
            model = load_saved_model(SAVE_DIR)
            if model is None:
                print("加载失败，将重新构建和训练模型。")
                model = build_model(X_train.shape[1])
                if model:
                     history, model = compile_and_train(model, X_train, y_train, X_val, y_val, SAVE_DIR, continue_training=False)
            else:
                 try:
                    if not hasattr(model, 'optimizer') or model.optimizer is None:
                        print("加载的模型缺少优化器状态，将重新编译。")
                        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
                        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
                    else:
                        print("继续使用加载模型的优化器状态进行训练。")
                        current_lr = tf.keras.backend.get_value(model.optimizer.learning_rate)
                        new_lr = min(current_lr * 0.7, 0.0005)

                        # --- 修复应用点 ---
                        if hasattr(model.optimizer.learning_rate, 'assign'):
                            model.optimizer.learning_rate.assign(new_lr)
                        else:
                            model.optimizer.learning_rate = new_lr
                        # --- 结束修复 ---

                        updated_lr = tf.keras.backend.get_value(model.optimizer.learning_rate)
                        print(f"继续训练的学习率设置为: {updated_lr:.6f}")

                    # Proceed with training only if model is valid
                    if model:
                        history, model = compile_and_train(model, X_train, y_train, X_val, y_val, SAVE_DIR, continue_training=True)

                 except Exception as e:
                     print(f"尝试继续训练时出错: {e}")
                     traceback.print_exc()
                     # Model might be in an unusable state
                     print("无法继续训练，请尝试直接测试或删除旧模型重新开始。")
                     # Keep the loaded model for potential testing? Or set to None?
                     # Setting to None might be safer if state is corrupted.
                     # model = None # Or just let it proceed to evaluation/GUI

        else:
            print("无效选择，将加载模型并直接测试。")
            model = load_saved_model(SAVE_DIR)
            if model is None:
                 print("加载失败，将重新构建和训练模型。")
                 model = build_model(X_train.shape[1])
                 if model:
                      history, model = compile_and_train(model, X_train, y_train, X_val, y_val, SAVE_DIR, continue_training=False)
    else:
        print(f"未在 {SAVE_DIR} 找到模型，将构建新模型并训练。")
        model = build_model(X_train.shape[1])
        if model: # Check if build succeeded
             history, model = compile_and_train(model, X_train, y_train, X_val, y_val, SAVE_DIR, continue_training=False)

    # --- Final Check and Exit if Model Failed ---
    if model is None:
        print("错误：模型未能成功加载或训练。无法继续。退出。")
        sys.exit(1) # Exit the script if no model is available

    # --- Evaluation ---
    eval_start_time = time.time()
    evaluate_model(model, X_test, y_test) # Pass the definitely non-None model
    print(f"模型评估耗时: {time.time() - eval_start_time:.2f} 秒")

    # --- Plotting ---
    if history: # Check if training actually happened and produced history
        plot_start_time = time.time()
        try:
            plot_training_history(history)
        except Exception as e:
            print(f"绘图失败: {e}。")
            # traceback.print_exc() # Uncomment for detailed plot errors
        print(f"绘制图表耗时: {time.time() - plot_start_time:.2f} 秒")

    # --- Launch GUI ---
    print("\n--- 启动绘图应用程序 ---")
    total_time_before_gui = time.time() - overall_start_time
    print(f"启动GUI前的总耗时: {total_time_before_gui:.2f} 秒")

    root = tk.Tk()
    try:
        s = ttk.Style()
        try:
            s.theme_use('vista') # Try preferred theme
        except tk.TclError:
            try: s.theme_use('clam') # Fallback theme
            except tk.TclError: print("警告: 未能设置 'vista' 或 'clam' ttk 主题。")
        app = DrawingApp(root, model) # Pass the non-None model
        if hasattr(app, 'main_frame'): # Check if init was successful before mainloop
            root.mainloop()
        else:
            print("GUI 初始化失败，无法启动 mainloop。")

    except Exception as e:
        print(f"GUI运行时发生严重错误: {e}")
        traceback.print_exc()

    print("\n--- 脚本执行完毕 ---")
    print(f"脚本总运行时间 (包括GUI关闭后): {time.time() - overall_start_time:.2f} 秒")


if __name__ == "__main__":
    main()