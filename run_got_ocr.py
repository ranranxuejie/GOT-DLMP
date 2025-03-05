
import subprocess

img_path = 'datasets/test_img/DLMP046.jpg'
type = 'format'
box = False
color = False
render = True
multi_page = False
cmd = [
    "python",
    "GOT/demo/run_ocr_2.0.py",
    "--model-name", 'GOT_weights/',
    "--image-file", img_path,
    "--type", type,
]
if render:
    cmd.append('--render')

if box:
    cmd.append('--box')
    cmd.append('[x1,y1,x2,y2]')
if color:
    cmd.append('--color')
    cmd.append('red/green/blue')
if multi_page:
    img_path = '/'.join(img_path.split('/')[:-1])
    cmd=[
        "python",
        "GOT/demo/run_ocr_2.0_crop.py",
        "--model-name", 'GOT_weights/',
        "--image-file", img_path,
        "--multi-page"
    ]
# 执行命令
subprocess.run(cmd, check=True)
