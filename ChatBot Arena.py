import requests
from bs4 import BeautifulSoup
import os
import webbrowser
from datetime import datetime
import logging
import atexit

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chatbot Arena的URL
URL = "https://openlm.ai/chatbot-arena/"

# 请求头，模拟浏览器访问
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

# 模型官网链接字典
MODEL_URLS = {
    "DeepSeek-R1": "https://www.deepseek.com/",
    "DeepSeek-V3": "https://www.deepseek.com/",
    "Qwen2.5-72B-Instruct": "https://qwenlm.github.io/",
    "Llama-3.3-70B-Instruct": "https://llama.meta.ai/",
    "Grok-3-Preview-02-24": "https://xai.company/",
    "GPT-4.5-Preview": "https://openai.com/",
    "Gemini-2.0-Flash-Thinking-Exp-01-21": "https://deepmind.google/",
    "Deepseek-v2.5-1210": "https://www.deepseek.com/",
    "Athene-v2-Chat-72B": "https://nexusflow.ai/",
    "GPT-4o-mini-2024-07-18": "https://openai.com/"
}

# 组织官网链接字典
ORG_URLS = {
    "DeepSeek": "https://www.deepseek.com/",
    "Qwen": "https://qwenlm.github.io/",
    "Meta AI": "https://llama.meta.ai/",
    "xAI": "https://xai.company/",
    "OpenAI": "https://openai.com/",
    "Google": "https://deepmind.google/",
    "NexusFlow": "https://nexusflow.ai/"
}

def get_text_width(text):
    """计算文本宽度，中文字符算2个单位，英文算1个单位，奖牌符号算2个单位"""
    width = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符
            width += 2
        elif char in ['🥇', '🥈', '🥉']:  # 奖牌符号
            width += 2
        else:
            width += 1
    return width

def fetch_chatbot_arena_data():
    """从Chatbot Arena网页抓取排行榜数据，返回控制台文本和HTML内容"""
    try:
        session = requests.Session()
        response = session.get(URL, headers=HEADERS, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table', class_='sortable')
        if not tables:
            logging.warning("未找到排行榜表格")
            return "无法找到排行榜表格，请检查网页结构是否变更。", "<p>无法找到排行榜表格，请检查网页结构是否变更。</p>"

        console_output = []
        html_output = ['<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Chatbot Arena 排行榜</title></head><body>',
                       '<style>table {border-collapse: collapse;} th, td {border: 1px solid black; padding: 8px; text-align: left;}</style>']

        headers_best_cn = ["模型", "竞技场Elo", "MMLU", "许可证"]
        headers_full_cn = ["模型", "竞技场Elo", "编码", "视觉", "竞技场难度", "MMLU", "投票数", "组织", "许可证"]

        # 计算列宽（仅用于控制台）
        column_widths_best = [40, 12, 10, 15]
        best_rows = []
        if len(tables) >= 1:
            for row in tables[0].find('tbody').find_all('tr'):
                cells = [td.get_text(strip=True) for td in row.find_all('td')]
                best_rows.append(cells)
                for i, cell in enumerate(cells):
                    column_widths_best[i] = max(column_widths_best[i], get_text_width(cell))
            for i, header in enumerate(headers_best_cn):
                column_widths_best[i] = max(column_widths_best[i], get_text_width(header))

        column_widths_full = [40, 12, 10, 10, 12, 10, 12, 15, 15]
        full_rows = []
        if len(tables) >= 2:
            for row in tables[1].find('tbody').find_all('tr'):
                cells = [td.get_text(strip=True) for td in row.find_all('td')]
                full_rows.append(cells)
                for i, cell in enumerate(cells):
                    column_widths_full[i] = max(column_widths_full[i], get_text_width(cell))
            for i, header in enumerate(headers_full_cn):
                column_widths_full[i] = max(column_widths_full[i], get_text_width(header))

        # 控制台输出 - "Best Open LM" 表格
        if len(tables) >= 1:
            console_output.append("最佳开源模型:")
            total_width_best = sum(column_widths_best) + 3 * (len(column_widths_best) - 1) + 1
            console_output.append("-" * total_width_best)
            headers = [headers_best_cn[i].ljust(column_widths_best[i]) for i in range(len(headers_best_cn))]
            console_output.append(" | ".join(headers))
            console_output.append("-" * total_width_best)
            for cells in best_rows:
                formatted_cells = [cell.ljust(column_widths_best[i]) for i, cell in enumerate(cells)]
                console_output.append(" | ".join(formatted_cells))

        # HTML 输出 - "Best Open LM" 表格
        if len(tables) >= 1:
            html_output.append("<h2>最佳开源模型:</h2>")
            html_output.append("<table>")
            html_output.append("<tr>" + "".join(f"<th>{header}</th>" for header in headers_best_cn) + "</tr>")
            for cells in best_rows:
                model_name = cells[0]
                model_link = MODEL_URLS.get(model_name, "#")
                cells[0] = f'<a href="{model_link}" target="_blank">{model_name}</a>'
                html_output.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
            html_output.append("</table>")

        # 控制台输出 - "Full Leaderboard" 表格
        if len(tables) >= 2:
            console_output.append("\n完整排行榜:")
            total_width_full = sum(column_widths_full) + 3 * (len(column_widths_full) - 1) + 1
            console_output.append("-" * total_width_full)
            headers = [headers_full_cn[i].ljust(column_widths_full[i]) for i in range(len(headers_full_cn))]
            console_output.append(" | ".join(headers))
            console_output.append("-" * total_width_full)
            for cells in full_rows:
                formatted_cells = [cell.ljust(column_widths_full[i]) for i, cell in enumerate(cells)]
                model_text = cells[0]
                if any(marker in model_text for marker in ['🥇', '🥈', '🥉']):
                    console_output.append(" | ".join(formatted_cells))
                else:
                    console_output.append(" " + " | ".join(formatted_cells))

        # HTML 输出 - "Full Leaderboard" 表格
        if len(tables) >= 2:
            html_output.append("<h2>完整排行榜:</h2>")
            html_output.append("<table>")
            html_output.append("<tr>" + "".join(f"<th>{header}</th>" for header in headers_full_cn) + "</tr>")
            for cells in full_rows:
                model_name = cells[0]
                model_link = MODEL_URLS.get(model_name, "#")
                org_name = cells[-2]  # 倒数第二列是“组织”
                org_link = ORG_URLS.get(org_name, "#")
                cells[0] = f'<a href="{model_link}" target="_blank">{model_name}</a>'
                cells[-2] = f'<a href="{org_link}" target="_blank">{org_name}</a>'
                html_output.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
            html_output.append("</table>")

        html_output.append("</body></html>")
        return "\n".join(console_output), "\n".join(html_output)

    except requests.RequestException as e:
        logging.error(f"获取数据失败: {e}")
        return f"错误: {e}", f"<p>错误: {e}</p>"
    except Exception as e:
        logging.error(f"解析错误: {e}")
        return f"解析错误: {e}", f"<p>解析错误: {e}</p>"

def save_and_open_html(html_content):
    """保存HTML内容到文件并用Edge浏览器打开，关闭后删除"""
    filename = "chatbot_arena_temp.html"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    edge_path = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
    webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))
    webbrowser.get('edge').open(f"file://{os.path.abspath(filename)}")

    atexit.register(lambda: os.remove(filename) if os.path.exists(filename) else None)

def main():
    """主函数，执行一次操作"""
    logging.info(f"Chatbot Arena 数据抓取 - 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    console_data, html_data = fetch_chatbot_arena_data()

    # 打印控制台信息
    print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Chatbot Arena 内容 (来源: https://openlm.ai/chatbot-arena/):")
    print("-" * 80)
    print(console_data)
    print("-" * 80)

    # 生成并打开HTML文件
    save_and_open_html(html_data)
    print("HTML 文件已生成并在 Edge 浏览器中打开，关闭程序后文件将自动删除。")

if __name__ == "__main__":
    main()