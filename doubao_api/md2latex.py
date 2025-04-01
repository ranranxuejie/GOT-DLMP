def md_to_latex_table(md_table):
    lines = md_table.strip().split('\n')
    # 移除表头分隔线
    if len(lines) > 1 and set(lines[1]) <= {'-', '|', ':'}:
        del lines[1]

    num_columns = len(lines[0].strip().split('|')) - 2
    column_spec = '|'.join(['c'] * num_columns)
    latex_table = "\\begin{tabular}{|" + column_spec + "|}\n"
    latex_table += "\\hline\n"

    for line in lines:
        cells = [cell.strip() for cell in line.strip().split('|')[1:-1]]
        latex_row = ' & '.join(cells) + " \\\\"
        latex_table += latex_row + "\n"
        latex_table += "\\hline\n"

    latex_table += "\\end{tabular}"
    return latex_table
def template(info):
    response,image_path = info
    return {"image": image_path,
            "conversations":[
                {"from":"human",
                 "value":"<image>\nOCR_DLMP:"},
                {"from":"gpt",
                 "value":response}
            ]}
if __name__ == '__main__':

    # 示例 Markdown 表格
    md_table = """
    |项目|详情|
    | ---- | ---- |
    |产品型号|S11 - M.RL/10|
    |标准代号|GB1094.1 - Z - 1996 GB1094.3 - 85|
    |额定容量|（未清晰显示具体数值）KVA|
    |产品代号|1LT.710|
    |额定频率|50 Hz|
    |相数|3|
    |额定电压|10000 / 400 V|
    |短路阻抗|（未清晰显示具体数值）%|
    |绝缘水平|L1 75 AC 35 / AC 5|
    |联结组标号|Yyn0|
    |开关位置|高压侧|低压侧|
    | |电压(V)|电流(A)|电压(V)|电流(A)|
    |1|10500| |400| |
    |2|10000| |400| |
    |3|9500| | | |
    |冷却方式|ONAN|
    |使用条件|户外|
    |器身重|（未清晰显示具体数值）Kg|
    |油重|（未清晰显示具体数值）Kg|
    |总重|（未清晰显示具体数值）Kg|
    |出厂序号|（未清晰显示具体数值）|
    |生产公司|广西柳州特种变压器有限责任公司|"""

    latex_table = md_to_latex_table(md_table)
    print(latex_table)
