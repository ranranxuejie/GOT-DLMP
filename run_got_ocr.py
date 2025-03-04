
import subprocess
cmd = [
    "python",
    "GOT/demo/run_ocr_2.0.py",
    "--model-name", 'GOT_weights/',
    "--image-file", 'datasets/DLMP/org_imgs/DLMP003.jpg',
    "--type", 'format',
    "--render"
]

# 执行命令
subprocess.run(cmd, check=True)
#
