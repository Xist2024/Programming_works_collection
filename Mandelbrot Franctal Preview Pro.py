import pygame
import numpy as np
from numba import njit, prange
import os
from pathlib import Path
import argparse
import time
import sys
import random

# --- Pygame窗口置顶 (Windows特定) ---
IS_WINDOWS = sys.platform == 'win32'
if IS_WINDOWS:
    try:
        import win32gui
        import win32con

        PYWIN32_AVAILABLE = True
    except ImportError:
        PYWIN32_AVAILABLE = False
else:
    PYWIN32_AVAILABLE = False


# ========== 核心配置 ==========
APP_CONFIG = {
    "INIT_WIDTH": 1280, "INIT_HEIGHT": 720,
    "WINDOW_TITLE_BASE": "分形浏览器专业版", "FONT_SIZE": 16,  # 修改标题为中文
}
ITERATION_PARAMS = {
    "MIN_ADAPTIVE": 150, "MAX_ADAPTIVE_CAP": 8000,
    "ADAPTIVE_BASE": 300, "ADAPTIVE_SENSITIVITY": 150,
}
CONTROL_PARAMS = {
    "ZOOM_FACTOR": 1.25, "FINE_ZOOM_FACTOR_MODIFIER": 1.05,
    "BASE_MOVE_SPEED": 0.1, "FINE_MOVE_SPEED_MODIFIER": 0.25,
}

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "mandelbrot_config.npz"

DEBUG_TEXT_COLOR = (200, 220, 255);  # Semicolon here is fine, just not typical Python style
DEBUG_BG_COLOR = (15, 25, 55, 220)

FRACTAL_TYPES = {  # 只保留Mandelbrot和Julia
    1: "MandelbrotSet", 2: "JuliaSet",
}
COLOR_SCHEMES_MJ = {1: "Original", 2: "Grayscale", 3: "Fiery", }
DEFAULT_COLOR_SCHEME_MJ_ID = 1


# ====== 视图状态容器 ======
class ViewState:
    def __init__(self):
        self.center = np.array([-0.5, 0.0], dtype=np.float64)
        self.scale = 2.0
        self.initial_scale = self.scale
        self.julia_c = complex(-0.7, 0.27015)
        self.screen_size = (APP_CONFIG["INIT_WIDTH"], APP_CONFIG["INIT_HEIGHT"])
        self.show_debug = True
        self.cached_image = None
        self.cached_image_params = {}
        self.current_fractal_type_id = 1
        self.current_color_scheme_mj_id = DEFAULT_COLOR_SCHEME_MJ_ID
        # self.ifs_iterations = IFS_DEFAULT_ITERATIONS.copy() # 删除IFS相关
        # logger.debug("ViewState initialized.") # 删除日志

    def clear_render_cache(self):
        self.cached_image = None
        self.cached_image_params = {}
        # logger.debug("Render cache (image and params) cleared for Mandelbrot/Julia.") # 删除日志

    def get_magnification(self):
        return self.initial_scale / max(self.scale, 1e-16)


state = ViewState()


# ====== 配置文件操作 ======
def save_config(current_max_iter_mj):
    # logger.info(f"Attempting to save configuration to {CONFIG_PATH}") # 删除日志
    print(f"尝试保存配置至 {CONFIG_PATH}")  # 改为中文print
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {
            'center_x': state.center[0], 'center_y': state.center[1], 'scale': state.scale,
            'screen_w': state.screen_size[0], 'screen_h': state.screen_size[1],
            'color_scheme_mj_id': state.current_color_scheme_mj_id,
            'initial_scale': state.initial_scale,
            'julia_c_real': state.julia_c.real, 'julia_c_imag': state.julia_c.imag,
            'last_fractal_type_id': state.current_fractal_type_id,
        }
        # No IFS iterations to save

        cache_p = state.cached_image_params
        if (state.current_fractal_type_id in [1, 2] and
                state.cached_image is not None and cache_p and
                np.allclose(cache_p.get('center', np.array([np.nan, np.nan])), state.center, atol=1e-9) and
                np.isclose(cache_p.get('scale', np.nan), state.scale, atol=1e-9) and
                cache_p.get('size') == state.screen_size and
                np.isclose(cache_p.get('max_iter', -1.0), current_max_iter_mj, atol=1e-9) and
                cache_p.get('color_scheme_id') == state.current_color_scheme_mj_id and
                (state.current_fractal_type_id != 2 or
                 (np.isclose(cache_p.get('julia_c_real', np.nan), state.julia_c.real, atol=1e-9) and
                  np.isclose(cache_p.get('julia_c_imag', np.nan), state.julia_c.imag, atol=1e-9)))
        ):
            data_to_save['img_data_mj'] = state.cached_image
            data_to_save['max_iter_val_mj'] = current_max_iter_mj
            # logger.info("Valid M/J memory cache found, saving image data.") # 删除日志
            print("有效的M/J内存缓存已找到，正在保存图像数据。")  # 改为中文print
        else:
            # logger.info("No valid M/J memory cache for current state; M/J image data not saved.") # 删除日志
            print("当前状态没有有效的M/J内存缓存；M/J图像数据未保存。")  # 改为中文print
            if 'img_data_mj' in data_to_save: del data_to_save['img_data_mj']
            if 'max_iter_val_mj' in data_to_save: del data_to_save['max_iter_val_mj']

        np.savez_compressed(CONFIG_PATH, **data_to_save)
        # logger.info(f"Configuration successfully saved to {CONFIG_PATH}") # 删除日志
        print(f"配置已成功保存至 {CONFIG_PATH}")  # 改为中文print
    except Exception as e:
        # logger.error(f"Failed to save configuration: {e}", exc_info=True) # 删除日志
        print(f"保存配置失败: {e}")  # 改为中文print


def load_initial_config_params():
    # logger.info(f"Attempting to load initial configuration from {CONFIG_PATH}") # 删除日志
    print(f"尝试从 {CONFIG_PATH} 加载初始配置...")  # 改为中文print
    try:
        if not CONFIG_PATH.exists():
            # logger.info("Config file not found. Using default settings."); # 删除日志
            print("配置文件未找到。使用默认设置。")  # 改为中文print
            return False
        with np.load(CONFIG_PATH, allow_pickle=False) as data:
            state.center = np.array([data['center_x'], data['center_y']], dtype=np.float64)
            state.scale = float(data['scale'])
            state.initial_scale = float(data.get('initial_scale', state.scale))
            if 'screen_w' in data and 'screen_h' in data:
                state.screen_size = (max(int(data['screen_w']), 320), max(int(data['screen_h']), 240))
            state.current_color_scheme_mj_id = int(data.get('color_scheme_mj_id', DEFAULT_COLOR_SCHEME_MJ_ID))
            state.julia_c = complex(float(data.get('julia_c_real', state.julia_c.real)),
                                    float(data.get('julia_c_imag', state.julia_c.imag)))
            # Ensure loaded fractal type is valid (Mandelbrot or Julia)
            loaded_fractal_type_id = int(data.get('last_fractal_type_id', 1))
            if loaded_fractal_type_id not in FRACTAL_TYPES:  # Check against new, smaller FRACTAL_TYPES
                print(f"配置文件中的分形类型ID {loaded_fractal_type_id} 无效，将使用默认MandelbrotSet。")
                state.current_fractal_type_id = 1
            else:
                state.current_fractal_type_id = loaded_fractal_type_id
            # No IFS iterations to load

            # logger.info("View parameters loaded from config.") # 删除日志
            print("视图参数已从配置文件加载。")  # 改为中文print
            return True
    except Exception as e:
        # logger.error(f"Error loading config parameters: {e}", exc_info=True) # 删除日志
        print(f"加载配置参数时出错: {e}")  # 改为中文print
    return False


def load_mj_cached_image_from_disk(target_center, target_scale, target_screen_size, target_max_iter,
                                   target_color_scheme_id, target_julia_c=None):
    # logger.debug(f"Attempting M/J disk cache load for scale={target_scale:.3e}, iter={target_max_iter}") # 删除日志
    try:
        if not CONFIG_PATH.exists(): return None
        with np.load(CONFIG_PATH, allow_pickle=False) as data_check:
            required_mj_keys = ['center_x', 'center_y', 'scale', 'screen_w', 'screen_h',
                                'max_iter_val_mj', 'img_data_mj', 'color_scheme_mj_id']
            if target_julia_c is not None: required_mj_keys.extend(['julia_c_real', 'julia_c_imag'])
            if not all(key in data_check for key in required_mj_keys):
                # logger.debug("M/J disk config missing required cache keys."); # 删除日志
                return None

        with np.load(CONFIG_PATH, allow_pickle=False) as data:
            cached_center_mj = np.array([data['center_x'], data['center_y']], dtype=np.float64)
            cached_scale_mj = float(data['scale'])
            saved_max_iter_mj = int(data['max_iter_val_mj'])
            saved_cs_id_mj = int(data.get('color_scheme_mj_id', DEFAULT_COLOR_SCHEME_MJ_ID))
            img_data_mj_loaded = data['img_data_mj']

            params_match_mj = (
                    np.allclose(cached_center_mj, target_center, atol=1e-9) and
                    np.isclose(cached_scale_mj, target_scale, atol=1e-9) and
                    saved_max_iter_mj == target_max_iter and
                    saved_cs_id_mj == target_color_scheme_id
            )
            if target_julia_c is not None:
                params_match_mj = params_match_mj and \
                                  np.isclose(float(data.get('julia_c_real', np.nan)), target_julia_c.real,
                                             atol=1e-9) and \
                                  np.isclose(float(data.get('julia_c_imag', np.nan)), target_julia_c.imag, atol=1e-9)

            if params_match_mj:
                expected_shape_mj = (target_screen_size[1], target_screen_size[0], 3)
                if img_data_mj_loaded.shape == expected_shape_mj:
                    # logger.info("Matching M/J cached image found on disk and loaded.") # 删除日志
                    print("从磁盘加载了匹配的M/J缓存图像。")  # 改为中文print
                    return img_data_mj_loaded
                else:
                    # logger.warning(f"M/J Disk cache image shape mismatch. Expected {expected_shape_mj}, got {img_data_mj_loaded.shape}.") # 删除日志
                    print(
                        f"M/J磁盘缓存图像形状不匹配。预期 {expected_shape_mj}, 得到 {img_data_mj_loaded.shape}。")  # 改为中文print
                    return None
            else:
                # logger.debug("M/J disk cache parameters do not match target.") # 删除日志
                return None
    except Exception as e:
        # logger.error(f"Error loading M/J cached image from disk: {e}", exc_info=True) # 删除日志
        print(f"从磁盘加载M/J缓存图像时出错: {e}")  # 改为中文print
    return None


# ====== MANDELBROT/JULIA Iteration and Color ======
@njit(cache=True)
def calculate_mj_adaptive_max_iter(scale_val_param):
    eff_scale_calc = max(scale_val_param, 1e-16)
    adaptive_base_const = 300
    adaptive_sensitivity_const = 150
    min_adaptive_const = 150
    max_adaptive_cap_const = 8000
    val_iter_calc = adaptive_base_const + adaptive_sensitivity_const * abs(np.log10(eff_scale_calc))
    return int(max(min_adaptive_const, min(val_iter_calc, max_adaptive_cap_const)))


@njit(cache=True)
def hsv_to_rgb_mj(h_norm_param, s_norm_param, v_norm_param):
    h_int_conv = int((h_norm_param % 1.0) * 255.0);
    s_int_conv = int(max(0.0, min(1.0, s_norm_param)) * 255.0);
    v_int_conv = int(max(0.0, min(1.0, v_norm_param)) * 255.0)
    c_conv, h_float_conv, x_conv, m_conv = float(v_int_conv) * float(s_int_conv) / 255.0, float(h_int_conv), 0.0, 0.0
    x_conv = c_conv * (1.0 - abs((h_float_conv / 42.5) % 2.0 - 1.0));
    m_conv = float(v_int_conv) - c_conv
    r_out, g_out, b_out = 0.0, 0.0, 0.0
    if h_float_conv < 42.5:
        r_out, g_out, b_out = (c_conv + m_conv), (x_conv + m_conv), m_conv
    elif h_float_conv < 85.0:
        r_out, g_out, b_out = (x_conv + m_conv), (c_conv + m_conv), m_conv
    elif h_float_conv < 127.5:
        r_out, g_out, b_out = m_conv, (c_conv + m_conv), (x_conv + m_conv)
    elif h_float_conv < 170.0:
        r_out, g_out, b_out = m_conv, (x_conv + m_conv), (c_conv + m_conv)
    elif h_float_conv < 212.5:
        r_out, g_out, b_out = (x_conv + m_conv), m_conv, (c_conv + m_conv)
    else:
        r_out, g_out, b_out = (c_conv + m_conv), m_conv, (x_conv + m_conv)
    return (int(max(0.0, min(255.0, r_out))), int(max(0.0, min(255.0, g_out))), int(max(0.0, min(255.0, b_out))))


@njit(cache=True)
def get_mj_color(norm_iter_param, escaped_param, scheme_id_param, iter_smooth_param, max_iter_param):
    hn_calc, sn_calc, vn_calc = 0.0, 0.0, 0.0
    if scheme_id_param == 1:
        hn_calc, sn_calc, vn_calc = norm_iter_param, 1.0, 1.0 if escaped_param else 0.0
    elif scheme_id_param == 2:
        hn_calc, sn_calc, vn_calc = 0.0, 0.0, 1.0 - norm_iter_param if escaped_param else 0.0
    elif scheme_id_param == 3:
        hn_calc, sn_calc, vn_calc = 0.0 + norm_iter_param * 0.1666, 0.9 + 0.1 * np.sin(
            iter_smooth_param * 0.2), 1.0 if escaped_param else 0.0
    else:
        hn_calc, sn_calc, vn_calc = norm_iter_param, 1.0, 1.0 if escaped_param else 0.0
    return hsv_to_rgb_mj(hn_calc, sn_calc, vn_calc)


@njit(parallel=True, cache=True)
def compute_mandelbrot_julia_frame(center_x_mj, center_y_mj, view_scale_y_mj, max_iter_val_mj,
                                   screen_w_mj, screen_h_mj, color_scheme_id_mj,
                                   is_julia_flag, julia_c_real_param, julia_c_imag_param):
    img_mj = np.zeros((screen_h_mj, screen_w_mj, 3), dtype=np.uint8)
    ar_mj = float(screen_w_mj) / float(screen_h_mj);
    vsx_mj = float(view_scale_y_mj) * ar_mj
    iw_mj, ih_mj = 1.0 / float(screen_w_mj), 1.0 / float(screen_h_mj)
    for yp_mj in prange(screen_h_mj):
        im0_mj = (float(yp_mj) * ih_mj - 0.5) * view_scale_y_mj + center_y_mj
        for xp_mj in prange(screen_w_mj):
            re0_mj = (float(xp_mj) * iw_mj - 0.5) * vsx_mj + center_x_mj
            c_re_iter, c_im_iter = (julia_c_real_param, julia_c_imag_param) if is_julia_flag else (re0_mj, im0_mj)
            z_re_iter, z_im_iter = (re0_mj, im0_mj) if is_julia_flag else (0.0, 0.0)
            iter_raw_mj = 0
            for i_loop_mj in range(max_iter_val_mj):
                zrs_mj, zis_mj = z_re_iter * z_re_iter, z_im_iter * z_im_iter
                if zrs_mj + zis_mj > 4.0: iter_raw_mj = i_loop_mj; break
                z_im_n_mj = 2.0 * z_re_iter * z_im_iter + c_im_iter;
                z_re_iter = zrs_mj - zis_mj + c_re_iter;
                z_im_iter = z_im_n_mj
            else:
                iter_raw_mj = max_iter_val_mj
            iter_s_mj = float(iter_raw_mj);
            esc_mj = iter_raw_mj < max_iter_val_mj
            if esc_mj:
                zn2_mj = z_re_iter * z_re_iter + z_im_iter * z_im_iter
                if zn2_mj > 1e-9:
                    lzn_mj = np.log(zn2_mj) / 2.0;
                    if lzn_mj > 1e-9: iter_s_mj = float(iter_raw_mj) + 1.0 - (np.log(lzn_mj) / np.log(2.0))
            iter_s_mj = max(0.0, iter_s_mj)
            norm_i_mj = min(iter_s_mj, float(max_iter_val_mj)) / float(max_iter_val_mj)
            r_res_mj, g_res_mj, b_res_mj = get_mj_color(norm_i_mj, esc_mj, color_scheme_id_mj, iter_s_mj,
                                                        max_iter_val_mj)
            img_mj[yp_mj, xp_mj, 0] = r_res_mj;
            img_mj[yp_mj, xp_mj, 1] = g_res_mj;
            img_mj[yp_mj, xp_mj, 2] = b_res_mj
    return img_mj


# ====== IFS FRACTAL ALGORITHMS (DELETED as per request) ======

# ====== Pygame Display Message & Window Topmost ======
def display_pg_message(screen_surface_disp, font_obj_disp, msg_text_disp, duration_val=0):
    if not pygame.display.get_init() or screen_surface_disp is None or font_obj_disp is None: return
    try:
        s_rect_disp = screen_surface_disp.get_rect();
        txt_s_disp = font_obj_disp.render(msg_text_disp, True, DEBUG_TEXT_COLOR, DEBUG_BG_COLOR[:3])
        txt_r_disp = txt_s_disp.get_rect(center=s_rect_disp.center);
        bg_r_disp = txt_r_disp.inflate(40, 20)
        bg_s_disp = pygame.Surface(bg_r_disp.size, pygame.SRCALPHA);
        bg_s_disp.fill(DEBUG_BG_COLOR)
        screen_surface_disp.blit(bg_s_disp, bg_r_disp);
        screen_surface_disp.blit(txt_s_disp, txt_r_disp);
        pygame.display.flip()
        if duration_val > 0: pygame.time.wait(duration_val)
    except pygame.error as e_disp:
        print(f"Pygame显示消息时出错: {e_disp}")


def set_pg_window_topmost(title_str_top):
    if IS_WINDOWS and PYWIN32_AVAILABLE:
        try:
            hwnd_top = win32gui.FindWindow(None, title_str_top)
            if hwnd_top:
                win32gui.SetWindowPos(hwnd_top, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                      win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
            # else: # Removed logging for non-found window
        except Exception as e_top:
            print(f"设置窗口顶层时出错: {e_top}")
    elif IS_WINDOWS and not PYWIN32_AVAILABLE:
        if not hasattr(set_pg_window_topmost, "warned_pywin32_once_flag"):
            print("pywin32 库未找到。在Windows上窗口置顶功能不可用。请尝试 'pip install pywin32'")
            setattr(set_pg_window_topmost, "warned_pywin32_once_flag", True)


# ====== MAIN PROGRAM LOOP ======
def main_program_loop(fixed_iterations_mj_override_val=None, startup_fractal_type_id_val=None):
    pygame.init()
    program_start_mono_time_main = time.monotonic()
    load_initial_config_params()

    if startup_fractal_type_id_val is not None:
        state.current_fractal_type_id = startup_fractal_type_id_val

    # Ensure current_fractal_type_id is valid after loading config and potential override
    if state.current_fractal_type_id not in FRACTAL_TYPES:
        print(f"启动时分形类型ID {state.current_fractal_type_id} 无效，将使用MandelbrotSet。")
        state.current_fractal_type_id = 1  # Default to MandelbrotSet

    current_title_main = f"{APP_CONFIG['WINDOW_TITLE_BASE']} - {FRACTAL_TYPES.get(state.current_fractal_type_id, '未知')}"
    screen_sfc_main = pygame.display.set_mode(state.screen_size, pygame.RESIZABLE)
    pygame.display.set_caption(current_title_main)
    set_pg_window_topmost(current_title_main)

    game_main_clock_obj = pygame.time.Clock()

    # --- FONT LOADING (MODIFIED BLOCK) ---
    font_main_size_val = APP_CONFIG["FONT_SIZE"]
    debug_main_font_obj = None
    message_main_font_obj = None
    font_file_absolute_path_str = r"C:\Python312\pythonProject\SourceHanSansSC-Regular-2.otf"
    font_file_path_obj = Path(font_file_absolute_path_str)

    try:
        if font_file_path_obj.exists():
            debug_main_font_obj = pygame.font.Font(font_file_absolute_path_str, font_main_size_val)
            message_main_font_obj = pygame.font.Font(font_file_absolute_path_str, font_main_size_val + 4)
            print(f"调试和消息字体使用: {font_file_path_obj.name}")
        else:
            print(f"指定的字体文件未找到: {font_file_absolute_path_str}。尝试系统默认字体。")
            debug_main_font_obj = pygame.font.Font(None, font_main_size_val + 2)
            message_main_font_obj = pygame.font.Font(None, font_main_size_val + 6)
            print("调试和消息字体使用: Pygame默认 (可能无法显示中文)")
    except pygame.error as e_font_load:
        print(f"加载指定字体时发生 Pygame 错误: {e_font_load}。尝试系统默认字体。")
        debug_main_font_obj = pygame.font.Font(None, font_main_size_val + 2)
        message_main_font_obj = pygame.font.Font(None, font_main_size_val + 6)
        print("调试和消息字体使用: Pygame默认 (可能无法显示中文)")
    except Exception as e_font_generic:
        print(f"加载字体时发生未知错误: {e_font_generic}。尝试系统默认字体。")
        debug_main_font_obj = pygame.font.Font(None, font_main_size_val + 2)
        message_main_font_obj = pygame.font.Font(None, font_main_size_val + 6)
        print("调试和消息字体使用: Pygame默认 (可能无法显示中文)")

    if debug_main_font_obj is None:
        debug_main_font_obj = pygame.font.Font(None, font_main_size_val + 2)
        print("警告: 调试字体未能成功加载，使用最终回退默认字体。")
    if message_main_font_obj is None:
        message_main_font_obj = pygame.font.Font(None, font_main_size_val + 6)
        print("警告: 消息字体未能成功加载，使用最终回退默认字体。")
    # --- END OF FONT LOADING MODIFICATION ---

    is_running_main_flag = True;
    needs_redraw_main_flag = True
    last_calc_duration_main_val = 0.0;
    cache_hit_status_main_str = "N/A"
    # ifs_surface_cache_obj = None # Removed
    # last_ifs_params_rendered_dict = {} # Removed

    print(f"\n--- {APP_CONFIG['WINDOW_TITLE_BASE']} 已启动 ---")
    print("操作: WASD移动, 滚轮缩放, F1调试信息, [1-2]切换分形, Ctrl+[1-3]颜色方案, R重绘, Q退出.")
    if fixed_iterations_mj_override_val is not None:
        print(f"M/J 固定迭代次数: {fixed_iterations_mj_override_val}")
    else:
        print("M/J 使用自适应迭代模式.")
    print("----------------------------------------------------")

    while is_running_main_flag:
        dt_sec_main_val = game_main_clock_obj.tick(60) / 1000.0;
        dt_sec_main_val = max(dt_sec_main_val, 1.0 / 120.0)

        current_mj_max_iter_calc = 0;
        iter_mode_mj_info_str = ""
        if fixed_iterations_mj_override_val is not None:
            current_mj_max_iter_calc = fixed_iterations_mj_override_val
            iter_mode_mj_info_str = f"固定: {current_mj_max_iter_calc}"
        else:
            current_mj_max_iter_calc = calculate_mj_adaptive_max_iter(state.scale)
            iter_mode_mj_info_str = f"自适应: {current_mj_max_iter_calc}"

        mouse_pg_pos_curr = pygame.mouse.get_pos()
        w_scr_curr, h_scr_curr = state.screen_size;
        mouse_cx_curr, mouse_cy_curr = 0.0, 0.0
        if w_scr_curr > 0 and h_scr_curr > 0:
            aspect_scr_curr = float(w_scr_curr) / float(h_scr_curr)
            mouse_cx_curr = state.center[0] + (
                    mouse_pg_pos_curr[0] / float(w_scr_curr) - 0.5) * state.scale * aspect_scr_curr
            mouse_cy_curr = state.center[1] + (mouse_pg_pos_curr[1] / float(h_scr_curr) - 0.5) * state.scale

        for evt_item in pygame.event.get():
            if evt_item.type == pygame.QUIT:
                is_running_main_flag = False
            elif evt_item.type == pygame.VIDEORESIZE:
                new_dims_resize = (max(evt_item.w, 320), max(evt_item.h, 240))
                state.screen_size = new_dims_resize;
                state.clear_render_cache()
                # ifs_surface_cache_obj = None; last_ifs_params_rendered_dict = {} # Removed
                screen_sfc_main = pygame.display.set_mode(state.screen_size, pygame.RESIZABLE)
                needs_redraw_main_flag = True;
                set_pg_window_topmost(current_title_main)
            elif evt_item.type == pygame.MOUSEBUTTONDOWN and state.current_fractal_type_id in [1, 2]:
                if evt_item.button == 4 or evt_item.button == 5:
                    is_fine_zoom = pygame.key.get_pressed()[pygame.K_LSHIFT] or pygame.key.get_pressed()[
                        pygame.K_RSHIFT]
                    zoom_base_val = CONTROL_PARAMS["ZOOM_FACTOR"];
                    zoom_mod_val = CONTROL_PARAMS["FINE_ZOOM_FACTOR_MODIFIER"]
                    actual_zoom_val = zoom_base_val if not is_fine_zoom else zoom_mod_val
                    eff_zoom_val = actual_zoom_val if evt_item.button == 4 else (1.0 / actual_zoom_val)
                    norm_mx_val = mouse_pg_pos_curr[0] / float(w_scr_curr) - 0.5;
                    norm_my_val = mouse_pg_pos_curr[1] / float(h_scr_curr) - 0.5
                    mrb_val = state.center[0] + norm_mx_val * state.scale * aspect_scr_curr;
                    mib_val = state.center[1] + norm_my_val * state.scale
                    state.scale /= eff_zoom_val;
                    state.scale = max(state.scale, 1e-15)
                    mra_val = state.center[0] + norm_mx_val * state.scale * aspect_scr_curr;
                    mia_val = state.center[1] + norm_my_val * state.scale
                    state.center[0] += mrb_val - mra_val;
                    state.center[1] += mib_val - mia_val
                    state.clear_render_cache();
                    needs_redraw_main_flag = True
            elif evt_item.type == pygame.KEYDOWN:
                if evt_item.key == pygame.K_F1:
                    state.show_debug = not state.show_debug
                elif evt_item.key == pygame.K_q:
                    save_config(current_mj_max_iter_calc)
                    is_running_main_flag = False
                elif evt_item.key == pygame.K_r:
                    state.clear_render_cache()
                    # ifs_surface_cache_obj = None; last_ifs_params_rendered_dict = {} # Removed
                    needs_redraw_main_flag = True;
                    print("强制重绘。")
                elif pygame.K_1 <= evt_item.key <= pygame.K_2:  # Fractal Type (Only 1 and 2)
                    new_fractal_id_key = evt_item.key - pygame.K_0
                    if new_fractal_id_key in FRACTAL_TYPES:
                        if state.current_fractal_type_id != new_fractal_id_key:
                            state.current_fractal_type_id = new_fractal_id_key
                            current_title_main = f"{APP_CONFIG['WINDOW_TITLE_BASE']} - {FRACTAL_TYPES.get(state.current_fractal_type_id, '未知')}"
                            pygame.display.set_caption(current_title_main)
                            state.clear_render_cache()
                            # ifs_surface_cache_obj = None; last_ifs_params_rendered_dict = {} # Removed
                            needs_redraw_main_flag = True
                            print(f"分形类型切换至: {FRACTAL_TYPES[new_fractal_id_key]}")
                            if new_fractal_id_key == 2 and message_main_font_obj and screen_sfc_main:
                                display_pg_message(screen_sfc_main, message_main_font_obj,
                                                   f"Julia Set. C = {state.julia_c.real:.3f}{state.julia_c.imag:+.3f}j. 按 Ctrl+J 更改C值。",
                                                   2500)
                elif (pygame.key.get_mods() & pygame.KMOD_LCTRL or pygame.key.get_mods() & pygame.KMOD_RCTRL) and (
                        pygame.K_1 <= evt_item.key <= pygame.K_3):  # M/J Color Ctrl + 1-3
                    if state.current_fractal_type_id in [1, 2]:
                        scheme_id_mj_key = evt_item.key - pygame.K_0
                        if scheme_id_mj_key in COLOR_SCHEMES_MJ:
                            if state.current_color_scheme_mj_id != scheme_id_mj_key:
                                state.current_color_scheme_mj_id = scheme_id_mj_key
                                state.clear_render_cache();
                                needs_redraw_main_flag = True
                                print(f"M/J 颜色方案切换至: {COLOR_SCHEMES_MJ[scheme_id_mj_key]}")
                elif (
                        pygame.key.get_mods() & pygame.KMOD_LCTRL or pygame.key.get_mods() & pygame.KMOD_RCTRL) and evt_item.key == pygame.K_j:
                    if state.current_fractal_type_id == 2:
                        if message_main_font_obj and screen_sfc_main: display_pg_message(screen_sfc_main,
                                                                                         message_main_font_obj,
                                                                                         "准备更改 Julia C 值...", 100)
                        print("\n--- 更改 Julia Set 参数 C ---")
                        try:
                            c_real_str = input(f"输入 C 的实部 (当前: {state.julia_c.real:.6f}): ").strip()
                            c_imag_str = input(f"输入 C 的虚部 (当前: {state.julia_c.imag:.6f}): ").strip()
                            new_c_real = float(c_real_str) if c_real_str else state.julia_c.real
                            new_c_imag = float(c_imag_str) if c_imag_str else state.julia_c.imag
                            state.julia_c = complex(new_c_real, new_c_imag)
                            state.clear_render_cache();
                            needs_redraw_main_flag = True
                            print(f"Julia C 值已更新为: {state.julia_c.real}{state.julia_c.imag:+}j")
                        except ValueError:
                            print("无效的C值输入。请确保输入数字。")
                            if message_main_font_obj and screen_sfc_main: display_pg_message(screen_sfc_main,
                                                                                             message_main_font_obj,
                                                                                             "无效的C值!", 1500)
                        except Exception as e_julia_c:
                            print(f"更改Julia C时出错: {e_julia_c}")
                            if message_main_font_obj and screen_sfc_main: display_pg_message(screen_sfc_main,
                                                                                             message_main_font_obj,
                                                                                             "更改C值出错!", 1500)
                        print("-----------------------------")

        if state.current_fractal_type_id in [1, 2]:
            pressed_keys_map_pan = pygame.key.get_pressed()
            is_fine_pan_active = pressed_keys_map_pan[pygame.K_LSHIFT] or pressed_keys_map_pan[pygame.K_RSHIFT]
            pan_x_val, pan_y_val = 0.0, 0.0
            if pressed_keys_map_pan[pygame.K_w]: pan_y_val -= 1.0
            if pressed_keys_map_pan[pygame.K_s]: pan_y_val += 1.0
            if pressed_keys_map_pan[pygame.K_a]: pan_x_val -= 1.0
            if pressed_keys_map_pan[pygame.K_d]: pan_x_val += 1.0
            if pan_x_val != 0 or pan_y_val != 0:
                pan_v_screen = np.array([pan_x_val, pan_y_val], dtype=np.float64)
                norm_pan_v = np.linalg.norm(pan_v_screen)
                if norm_pan_v > 0: pan_v_screen /= norm_pan_v
                speed_modifier_pan = CONTROL_PARAMS["FINE_MOVE_SPEED_MODIFIER"] if is_fine_pan_active else 1.0
                complex_move_pan = CONTROL_PARAMS[
                                       "BASE_MOVE_SPEED"] * state.scale * dt_sec_main_val * speed_modifier_pan
                state.center[0] += pan_v_screen[0] * complex_move_pan
                state.center[1] += pan_v_screen[1] * complex_move_pan
                state.clear_render_cache();
                needs_redraw_main_flag = True

        if needs_redraw_main_flag:
            w_render_curr, h_render_curr = state.screen_size;
            cache_hit_status_main_str = "否 (重新计算)"

            # Only Mandelbrot/Julia rendering path now
            is_mem_cache_mj_valid = False
            if state.cached_image is not None and state.cached_image_params:
                p_mj_cache = state.cached_image_params
                is_mem_cache_mj_valid = (
                        np.allclose(p_mj_cache.get('center', np.array([np.nan, np.nan])), state.center, atol=1e-9) and
                        np.isclose(p_mj_cache.get('scale', np.nan), state.scale, atol=1e-9) and
                        p_mj_cache.get('size') == (w_render_curr, h_render_curr) and
                        np.isclose(p_mj_cache.get('max_iter', -1.0), current_mj_max_iter_calc, atol=1e-9) and
                        p_mj_cache.get('color_scheme_id') == state.current_color_scheme_mj_id and
                        (state.current_fractal_type_id != 2 or
                         (np.isclose(p_mj_cache.get('julia_c_real', np.nan), state.julia_c.real, atol=1e-9) and
                          np.isclose(p_mj_cache.get('julia_c_imag', np.nan), state.julia_c.imag, atol=1e-9))))
            if is_mem_cache_mj_valid:
                cache_hit_status_main_str = "是 (内存)"
            else:
                mj_disk_cache_img = load_mj_cached_image_from_disk(
                    state.center, state.scale, (w_render_curr, h_render_curr),
                    current_mj_max_iter_calc, state.current_color_scheme_mj_id,
                    state.julia_c if state.current_fractal_type_id == 2 else None)
                if mj_disk_cache_img is not None:
                    state.cached_image = mj_disk_cache_img
                    cache_hit_status_main_str = "是 (磁盘)"
                    state.cached_image_params = {'center': state.center.copy(), 'scale': state.scale,
                                                 'size': (w_render_curr, h_render_curr),
                                                 'max_iter': current_mj_max_iter_calc,
                                                 'color_scheme_id': state.current_color_scheme_mj_id,
                                                 'julia_c_real': state.julia_c.real, 'julia_c_imag': state.julia_c.imag}
                else:
                    print(f"重新计算 {FRACTAL_TYPES.get(state.current_fractal_type_id, '分形')} 帧...")
                    if message_main_font_obj and screen_sfc_main: display_pg_message(screen_sfc_main,
                                                                                     message_main_font_obj,
                                                                                     "正在计算...", 0)
                    t_start_mj_calc = time.perf_counter()
                    state.cached_image = compute_mandelbrot_julia_frame(
                        state.center[0], state.center[1], state.scale, current_mj_max_iter_calc, w_render_curr,
                        h_render_curr,
                        state.current_color_scheme_mj_id, is_julia_flag=(state.current_fractal_type_id == 2),
                        julia_c_real_param=state.julia_c.real, julia_c_imag_param=state.julia_c.imag)
                    last_calc_duration_main_val = time.perf_counter() - t_start_mj_calc
                    print(f"M/J 计算耗时: {last_calc_duration_main_val:.3f}秒")
                    state.cached_image_params = {'center': state.center.copy(), 'scale': state.scale,
                                                 'size': (w_render_curr, h_render_curr),
                                                 'max_iter': current_mj_max_iter_calc,
                                                 'color_scheme_id': state.current_color_scheme_mj_id,
                                                 'julia_c_real': state.julia_c.real, 'julia_c_imag': state.julia_c.imag}
            if state.cached_image is not None and screen_sfc_main:
                try:
                    if state.cached_image.shape == (h_render_curr, w_render_curr, 3):
                        screen_sfc_main.blit(pygame.surfarray.make_surface(np.swapaxes(state.cached_image, 0, 1)),
                                             (0, 0))
                except Exception as e_mj_blit_render:
                    print(f"M/J Blit错误: {e_mj_blit_render}");
                    state.clear_render_cache()

            needs_redraw_main_flag = False

        if state.show_debug and debug_main_font_obj and screen_sfc_main:
            total_rt_main = time.monotonic() - program_start_mono_time_main
            f_type_name_disp = FRACTAL_TYPES.get(state.current_fractal_type_id, "未知")
            debug_txt_list_render = [f"FPS: {game_main_clock_obj.get_fps():.1f}",
                                     f"计算: {last_calc_duration_main_val:.3f}s",
                                     f"缓存: {cache_hit_status_main_str}", f"运行: {total_rt_main:.1f}s", "---",
                                     f"分形: {f_type_name_disp} ([1-2])"]
            debug_txt_list_render.extend([
                f"放大: {state.get_magnification():.2e}x", f"迭代: {iter_mode_mj_info_str}",
                f"颜色: {COLOR_SCHEMES_MJ.get(state.current_color_scheme_mj_id, '未知')} (Ctrl+[1-3])", "---",
                f"中心 Re: {state.center[0]:.14f}", f"中心 Im: {state.center[1]:.14f}",
                f"视图 Scale: {state.scale:.6e}",
                f"鼠标 Re: {mouse_cx_curr:.14f}", f"鼠标 Im: {mouse_cy_curr:.14f}"
            ])
            if state.current_fractal_type_id == 2:
                debug_txt_list_render.append(
                    f"Julia C: {state.julia_c.real:.4f}{state.julia_c.imag:+.4f}j (Ctrl+J更改)")

            debug_txt_list_render.extend(["---", f"窗口: {state.screen_size[0]}x{state.screen_size[1]}",
                                          "Q:退出 F1:调试 R:重绘"])

            text_surf_list_disp, max_w_disp, y_off_disp = [], 0, 0
            pad_disp, lh_disp = 10, debug_main_font_obj.get_height()
            for line_str_disp in debug_txt_list_render:
                if line_str_disp == "---": y_off_disp += lh_disp // 3; continue
                try:
                    s_disp_render = debug_main_font_obj.render(line_str_disp, True, DEBUG_TEXT_COLOR)
                    text_surf_list_disp.append({'s': s_disp_render, 'y': y_off_disp})
                    if s_disp_render.get_width() > max_w_disp: max_w_disp = s_disp_render.get_width()
                    y_off_disp += lh_disp + 2
                except pygame.error as e_font_disp:
                    print(f"渲染调试文本 '{line_str_disp}' 失败: {e_font_disp}")

            if text_surf_list_disp:
                panel_w_disp = min(max_w_disp + 2 * pad_disp, state.screen_size[0] - 16)
                panel_h_disp = y_off_disp - 2 + 2 * pad_disp
                debug_panel_sfc_render = pygame.Surface((panel_w_disp, panel_h_disp), pygame.SRCALPHA)
                debug_panel_sfc_render.fill(DEBUG_BG_COLOR)
                for item_disp_blit in text_surf_list_disp:
                    debug_panel_sfc_render.blit(item_disp_blit['s'], (pad_disp, pad_disp + item_disp_blit['y']))
                screen_sfc_main.blit(debug_panel_sfc_render, (8, 8))

        pygame.display.flip()

    pygame.quit()
    print(f"--- {APP_CONFIG['WINDOW_TITLE_BASE']} 已关闭 ---")


if __name__ == "__main__":
    print("应用程序启动中...")
    parser = argparse.ArgumentParser(description=f"{APP_CONFIG['WINDOW_TITLE_BASE']}.")
    parser.add_argument("-i", "--iterations", type=int, default=None, help="固定M/J迭代次数(可选)")
    cli_args_main = parser.parse_args()

    chosen_mj_iterations_cli = cli_args_main.iterations
    chosen_fractal_id_startup = None

    if chosen_mj_iterations_cli is not None and chosen_mj_iterations_cli <= 0:
        print("命令行指定的迭代次数必须为正。将使用自适应/默认值。")
        chosen_mj_iterations_cli = None

    print(f"\n欢迎使用 {APP_CONFIG['WINDOW_TITLE_BASE']}!")
    print("------------------------------------")
    print("请选择要探索的分形:")
    for fid_startup, fname_startup in FRACTAL_TYPES.items():
        print(f"  {fid_startup}. {fname_startup}")

    while chosen_fractal_id_startup is None:
        try:
            fractal_choice_input = input("请输入分形编号: ").strip()
            fid_choice_input = int(fractal_choice_input)
            if fid_choice_input in FRACTAL_TYPES:
                chosen_fractal_id_startup = fid_choice_input
                print(f"已选择分形: {FRACTAL_TYPES[chosen_fractal_id_startup]}")
            else:
                print(f"无效选择。请输入 {', '.join(map(str, FRACTAL_TYPES.keys()))} 中的一个。")  # 更明确的提示
        except ValueError:
            print("无效输入。请输入一个数字。")
        except EOFError:
            print("输入已取消。正在退出。");
            sys.exit()

    if chosen_mj_iterations_cli is None:
        while True:
            try:
                print("------------------------------------")
                user_input_iter_mj = input(
                    f"请输入 {FRACTAL_TYPES[chosen_fractal_id_startup]} 的迭代次数 (例如 300), 或按 Enter/输入0 使用自适应: ").strip()
                if not user_input_iter_mj or user_input_iter_mj == "0":
                    chosen_mj_iterations_cli = None;
                    print("将使用自适应迭代。");
                    break
                num_iter_mj_input = int(user_input_iter_mj)
                if num_iter_mj_input > 0:
                    chosen_mj_iterations_cli = num_iter_mj_input;
                    print(f"将使用固定迭代次数: {chosen_mj_iterations_cli}");
                    break
                else:
                    print("迭代次数必须为正整数。")
            except ValueError:
                print("无效输入。请输入一个整数。")
            except EOFError:
                print("\n输入已取消。将使用自适应迭代。");
                chosen_mj_iterations_cli = None;
                break
    elif chosen_mj_iterations_cli is not None:  # Only print if it was set by CLI and valid
        print(f"使用命令行指定的固定M/J迭代次数: {chosen_mj_iterations_cli}")

    if chosen_fractal_id_startup == 2:
        print("------------------------------------")
        print(f"当前 Julia C = {state.julia_c.real:.4f}{state.julia_c.imag:+.4f}j")
        use_default_c_input = input("是否使用当前/默认的 Julia C 值? (Y/n): ").strip().lower()
        if use_default_c_input == 'n':
            while True:
                try:
                    c_real_input = input("请输入 Julia Set C值的实部: ").strip()
                    c_imag_input = input("请输入 Julia Set C值的虚部: ").strip()
                    state.julia_c = complex(float(c_real_input), float(c_imag_input))
                    print(f"Julia C 已设置为: {state.julia_c.real}{state.julia_c.imag:+}j")
                    break
                except ValueError:
                    print("无效的C值输入。请输入数字。")
                except EOFError:
                    print("C值输入已取消。将使用当前/默认C值。");
                    break
    try:
        main_program_loop(fixed_iterations_mj_override_val=chosen_mj_iterations_cli,
                          startup_fractal_type_id_val=chosen_fractal_id_startup)
    except Exception as e_main_outer:
        print(f"应用程序发生未处理的严重错误: {e_main_outer}")
        import traceback

        traceback.print_exc()
        if pygame.get_init(): pygame.quit()
        sys.exit("应用程序因严重错误而终止。")
    print("应用程序已结束。")