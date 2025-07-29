import math
import sys
import os
import time
import threading

# 尝试导入gmpy2库
try:
    import gmpy2

    GMPY2_AVAILABLE = True
except ImportError:
    GMPY2_AVAILABLE = False

# 增加Python的递归深度限制
sys.setrecursionlimit(5000)

# --- 新增：您的完整经验数据库 ---
EMPIRICAL_DATA = [
    (25000000, 300, 89.0), (25000000, 150, 63.0), (25000000, 75, 46.0), (25000000, 25, 0.5),
    (10000000, 300, 36.0), (10000000, 150, 25.0), (10000000, 75, 18.0), (10000000, 25, 0.5),
    (5000000, 300, 19.0), (5000000, 150, 13.0), (5000000, 75, 10.0), (5000000, 25, 0.5),
]


def clear_console():
    """
    跨平台的清空控制台函数。
    """
    os.system('cls' if os.name == 'nt' else 'clear')


class UltimateSolver:
    """
    最终版定理求解器 (v14.0)。
    """

    def __init__(self, n, x):
        self.iterations = 0
        self.last_checked_iterations = 0
        self.current_status = "正在初始化..."

        if GMPY2_AVAILABLE:
            self.n = gmpy2.mpz(n)
            self.x = gmpy2.mpz(x)
        else:
            self.n = n
            self.x = x

        self.target_sum = self.x * (self.n ** 2)
        self.memo = {}

    def _get_sum_of_squares_up_to(self, k):
        if k <= 0:
            return 0
        k_val = gmpy2.mpz(k) if GMPY2_AVAILABLE else k
        self.current_status = f"正在计算 {len(str(k))}-位 数的累加和公式..."
        return k_val * (k_val + 1) * (2 * k_val + 1) // 6

    def solve(self):
        if GMPY2_AVAILABLE:
            self.current_status = "正在计算超大数平方根..."
            max_start_val = gmpy2.isqrt(self.target_sum)
        else:
            max_start_val = math.floor(math.sqrt(self.target_sum))

        return self._find_recursive(self.target_sum, max_start_val)

    def _find_recursive(self, target, max_val):
        self.iterations += 1
        self.current_status = "正在快速搜索解空间..."

        if target == 0:
            return []
        if target < 0 or max_val <= 0:
            return None

        state = (target, max_val)
        if state in self.memo:
            return self.memo[state]

        if target > self._get_sum_of_squares_up_to(max_val):
            self.memo[state] = None
            return None

        for i in range(max_val, 0, -1):
            self.current_status = f"正在计算 {len(str(i))}-位 数的平方..."
            current_square = i * i

            if current_square > target:
                continue

            self.current_status = "正在进行剪枝检查..."
            if (target - current_square) > self._get_sum_of_squares_up_to(i - 1):
                continue

            result = self._find_recursive(target - current_square, i - 1)
            if result is not None:
                solution = [i] + result
                self.memo[state] = solution
                return solution

        self.memo[state] = None
        return None


def display_status(solver, start_time, stop_event):
    """
    这个函数在后台线程中运行，用于动态显示状态。
    """
    while not stop_event.is_set():
        elapsed_time = time.perf_counter() - start_time
        if solver.iterations > solver.last_checked_iterations:
            status_line = f"\r[ 计算中... | 已耗时: {elapsed_time:.2f}s ]"
            solver.last_checked_iterations = solver.iterations
        else:
            status_line = f"\r[ {solver.current_status} | 已耗时: {elapsed_time:.2f}s ]"

        sys.stdout.write(status_line + " " * 30)
        sys.stdout.flush()
        time.sleep(0.1)


def display_header():
    """显示欢迎信息和使用说明。"""
    print("---------------------------------------------------------------------------------------------------------")
    print("本工具为一个定律寻找满足所有约束的整式实例")
    print("定律即:对于任意正整数n和x，总能找到一组互不相同，逐渐递减，的正整数，使其平方和除以n²的结果恰好等于x")
    print("计算完成后，按回车键可清空并进行下一次计算")
    print(f"加速引擎: {'gmpy2 (已启用)' if GMPY2_AVAILABLE else 'Python原生 (未找到gmpy2)'}")
    print("标准整式格式为: (a₁² + a₂² + ... + aₖ²) / n² = x")
    print("")
    print("换言之，这个工具可以求解以下特定形式的Diophantus方程")
    print("a₁² + a₂² + ... + aₖ² = x*n²")
    print("输出整式的分子即为方程的左半部分")
    print("---------------------------------------------------------------------------------------------------------")


def predict_time(n, x):
    """
    新增：使用线性插值法，根据用户的经验数据来预测时间。
    """
    if n < 1000000 and x <= 300:
        print("预计耗时: 最多3秒")
        return

    try:
        current_workload = n * math.sqrt(x)
    except (ValueError, TypeError):
        print("无法进行预测，输入值无效。")
        return

    # 预处理数据，计算每个数据点的工作量指标并排序
    processed_data = sorted([(dn * math.sqrt(dx), dt) for dn, dx, dt in EMPIRICAL_DATA])

    # --- 线性插值逻辑 ---
    lower_bound = (0, 0.5)  # 默认下界
    upper_bound = None

    # 寻找上下界邻居
    for workload, time_val in processed_data:
        if workload <= current_workload:
            lower_bound = (workload, time_val)
        else:
            upper_bound = (workload, time_val)
            break  # 找到第一个上界后即可停止

    # 进行预测
    if upper_bound is None:  # 如果新输入比所有数据都大，进行线性外插
        w1, t1 = processed_data[-2] if len(processed_data) > 1 else (0, 0.5)
        w2, t2 = processed_data[-1]
        if w2 == w1: w2 = w1 + 1  # 避免除零错误
        # 外插公式： y = y2 + (x - x2) * (y2 - y1) / (x2 - x1)
        predicted_time = t2 + (current_workload - w2) * (t2 - t1) / (w2 - w1)
    else:  # 否则，进行线性内插
        w1, t1 = lower_bound
        w2, t2 = upper_bound
        if w2 == w1: w2 = w1 + 1  # 避免除零错误
        # 内插公式: t = t1 + (t2 - t1) * (w - w1) / (w2 - w1)
        predicted_time = t1 + (t2 - t1) * (current_workload - w1) / (w2 - w1)

    # 对预测结果进行人性化处理
    if predicted_time < 1.0:
        print("预计耗时: 很快完成 (小于1秒)")
    else:
        print(f"预计耗时: 约 {max(1, int(round(predicted_time)))} 秒")


def main_loop():
    """
    程序的主循环。
    """
    while True:
        clear_console()
        display_header()
        try:
            n_input = int(input("\n请输入决定整式分母值的 n: "))
            x_input = int(input("请输入您期望得到的结果 x: "))
            predict_time(n_input, x_input)
            print("\n初始化求解器并开始求解...")

            solver = UltimateSolver(n_input, x_input)

            start_time = time.perf_counter()
            stop_event = threading.Event()
            status_thread = threading.Thread(target=display_status, args=(solver, start_time, stop_event))
            status_thread.start()

            solution = solver.solve()

            stop_event.set()
            status_thread.join()

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

            if solution:
                solution = [int(num) for num in solution]
                numerator_parts = [f"{val}²" for val in solution]
                numerator_str = " + ".join(numerator_parts)

                print("\n🎉 成功找到一组满足所有约束的解！ 🎉\n")
                print(f"({numerator_str})")
                line_len = max(len(numerator_str) + 2, 20)
                print("—" * line_len)
                print(f"{' ' * (line_len // 2 - 4)}{n_input}²")

                print(f"\n= {solver.target_sum} / {n_input ** 2}")
                print(f"= {x_input}")
            else:
                print("\n😔 抱歉，未能找到一组满足条件的解。")

            print(f"\n[本次求解用时: {elapsed_time:.6f} 秒]")

            input("\n输入回车再次计算：")

        except (ValueError, TypeError):
            print("\n输入无效，请输入正整数。将在2秒后重试...")
            time.sleep(2)
        except RecursionError:
            print("\n计算错误：问题过于复杂，导致递归深度超出限制。将在2秒后重试...")
            time.sleep(2)
        except KeyboardInterrupt:
            print("\n\n程序已由用户中断。再见！")
            sys.exit(0)


if __name__ == "__main__":
    main_loop()