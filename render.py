from markdown import markdown
import datetime
import imgkit
import json
result = json.load(open('./GOT/eval/result_checkpoint-8000-encoder.json', encoding='utf-8'))

current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
import os
os.environ["PATH"] += "C:\Program Files\wkhtmltopdf\\bin"

def latex_table_to_markdown(latex_table):
    """
    将LaTeX表格转换为Markdown格式
    参数:
        latex_table: str - 包含LaTeX表格的字符串
    返回:
        str - Markdown格式的表格
    """
    # 移除LaTeX表格环境声明和hline
    content = latex_table.replace('\\begin{tabular}', '') \
        .replace('\\end{tabular}', '') \
        .replace('\\hline', '') \
        .strip()

    # 使用正则表达式移除所有列格式定义（包括{|c|c|}、| c|}等）
    import re
    content = re.sub(r'\{?\s*\|[^}]*\|+\s*\}?', '', content)

    # 处理表格行
    rows = [row.strip() for row in content.split('\\\\') if row.strip()]

    # 处理每行中的单元格
    processed_rows = []
    for i, row in enumerate(rows):
        # 移除行首尾的空格和大括号
        row = re.sub(r'^\s*[{}]*\s*|\s*[{}]*\s*$', '', row)
        if not row:
            continue

        # 分割单元格
        cells = [cell.strip() for cell in row.split('&')]
        # 跳过纯分隔线行（如----）
        if all(re.match(r'^-+$', cell) for cell in cells):
            continue

        processed_rows.append('| ' + ' | '.join(cells) + ' |')

        # 添加表头分隔线（仅在第一行后添加）
        if i == 0:
            separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
            processed_rows.append(separator)

    return '\n'.join(processed_rows)
def md2html(md):
    html_table = markdown(md, extensions=['tables'])

    html_content = f"""
        <html>
        <head>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
        </style>
        </head>
        <body>
        {html_table}
        </body>
        </html>
        """
    return html_content
for img_name,outputs in result.items():
    if outputs.startswith('\\begin{tabular}'):
        markdown_table = latex_table_to_markdown(outputs)
        html_content = md2html(markdown_table)
        # 保存为HTML文件
        os.makedirs(f'./results/html/{current_time}',exist_ok=True)
        with open(f'./results/html/{current_time}/{img_name}.html', 'w', encoding='GB18030') as f:
            f.write(html_content)
