
import subprocess
#swift sft --model_type got-ocr2 --model_id_or_path ./GOT_weights --sft_type lora --dataset dlmp
# cmd = [
#     "swift",
#     "sft",
#     "--model_type", "got-ocr2",
#     "--model_id_or_path", "results/dlmp/checkpoint-8000",
#     "--sft_type", "lora",
#     "--dataset", "datasets/DLMP_got/DLMP_got_lora.jsonl",
# ]
# swift sft --model_type got-ocr2 --model_id_or_path./GOT_weights --sft_type lora --datasets/DLMP_got/DLMP_got_lora.jsonl

# subprocess.run(cmd, check=True)
#deepspeed   GOT/train/train_GOT.py \
 # --deepspeed zero_config/zero2.json    --model_name_or_path stepfun-ai/GOT-OCR2_0 \
 # --use_im_start_end True   \
 # --bf16 True   \
 # --gradient_checkpointing True \
 # --gradient_accumulation_steps 4    \
 # --evaluation_strategy "no"   \
 # --save_strategy "steps"  \
 # --save_steps 20   \
 # --save_total_limit 1   \
 # --weight_decay 0.    \
 # --warmup_ratio 0.001     \
 # --lr_scheduler_type "cosine"    \
 # --logging_steps 1    \
 # --tf32 True     \
 # --model_max_length 8192    \
 # --gradient_checkpointing True   \
 # --dataloader_num_workers 16    \
 # --report_to none  \
 # --per_device_train_batch_size 1    \
 # --num_train_epochs 10  \
 # --learning_rate 2e-5   \
 # --datasets dlmp \
 # --output_dir results/dlmp
cmd = [
    "deepspeed",
    "GOT/train/train_GOT.py",
    "--deepspeed", "zero_config/zero2.json",
    "--model_name_or_path", "stepfun-ai/GOT-OCR2_0",
    "--use_im_start_end", "True",
    "--bf16", "True",
    "--gradient_accumulation_steps", "2",
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "1000",
    "--save_total_limit", "1",
    "--weight_decay", "0.",
    "--warmup_ratio", "0.001",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--tf32", "True",
    "--model_max_length", "8192",
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "16",
    "--report_to", "none",
    "--per_device_train_batch_size", "2",
    "--num_train_epochs", "10",
    "--learning_rate", "2e-5",
    "--datasets", "dlmp",
    "--output_dir", "results/dlmp",
    "--resume_from_checkpoint", "results/dlmp"
]
subprocess.run(cmd, check=True)


# #%%ls
# img_path = 'datasets/test_img/DLMP046.jpg'
# type = 'format/ocr'
# box = False
# color = False
# render = False
# multi_page = True
# cmd = [
#     "python",
#     "GOT/demo/run_ocr_2.0.py",
#     "--model-name", 'GOT_weights/',
#     "--image-file", img_path,
#     "--type", type,
# ]
# if render:
#     cmd.append('--render')
#
# if box:
#     cmd.append('--box')
#     cmd.append('[x1,y1,x2,y2]')
# if color:
#     cmd.append('--color')
#     cmd.append('red/green/blue')
# if multi_page:
#     img_path = '/'.join(img_path.split('/')[:-1])
#     cmd=[
#         "python",
#         "GOT/demo/run_ocr_2.0_crop.py",
#         "--model-name", 'GOT_weights/',
#         "--image-file", img_path,
#         "--multi-page"
#     ]
# # 执行命令
# subprocess.run(cmd, check=True)
