"""
eval.py (vision_solver_main_mvc problem)

Entry point invoked by Hercules for each candidate individual. Reads the
candidate drawing function from gpt.py (written by Hercules just before
this script is launched), and computes its fitness by:
  1. Rendering a training image set with the candidate draw function.
  2. LoRA SFT fine-tuning a base VLM on those images.
  3. Evaluating the fine-tuned VLM on a fixed held-out eval image set
     (rendered with the SAME candidate function) against known optimal
     solutions, producing exact match and objective accuracy.

Hercules (hercules.py) reads this script's stdout and takes the SECOND TO
LAST line (since stdout ends with a trailing newline) as the objective
value, interpreted according to obj_type ("min" here). So the last thing
printed must be a bare float on its own line.

Usage (matches hercules.py._run_code convention):
    python eval.py <problem_size> <root_dir> <mode>

problem_size is interpreted as the number of training instances to render
(defaults to 2000 if not a sane positive int, e.g. when mode="val" passes -1).
"""

import os
import shutil
import subprocess
import sys
import time
import datetime

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _int_or(default, s):
    try:
        v = int(s)
        return v if v > 0 else default
    except Exception:
        return default


def main():
    args = sys.argv[1:]
    problem_size = _int_or(2000, args[0]) if len(args) > 0 else 2000
    root_dir = args[1] if len(args) > 1 else os.path.abspath(os.path.join(THIS_DIR, '..', '..'))
    mode = args[2] if len(args) > 2 else 'train'

    # Use the correct Python environment with unsloth
    py = '/root/autodl-tmp/yao/llm_env/bin/python'

    code_path = os.path.join(THIS_DIR, 'gpt.py')

    # MVC dataset paths
    train_json = os.environ.get(
        'MVC_TRAIN_JSON',
        os.path.join(THIS_DIR, 'train_mvc_under30_5000.json'),
    )
    eval_json = os.environ.get(
        'MVC_EVAL_JSON',
        os.path.join(THIS_DIR, 'test_mvc_under30_50.json'),
    )
    base_model = os.environ.get(
        'MVC_BASE_MODEL',
        '/root/autodl-tmp/yao/models1_cache/models/Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master',
    )
    num_train_samples = _int_or(problem_size, os.environ.get('MVC_NUM_TRAIN', str(problem_size)))
    max_steps = _int_or(-1, os.environ.get('MVC_MAX_STEPS', '-1'))
    num_train_epochs = _int_or(1, os.environ.get('MVC_EPOCHS', '1'))
    render_workers = _int_or(1, os.environ.get('MVC_RENDER_WORKERS', '1'))
    num_eval_samples = _int_or(50, os.environ.get('MVC_NUM_EVAL', '50'))

    run_tag = f"{datetime.datetime.now().strftime('%m%d%H%M')}_ID_{os.getpid()}"
    run_dir = os.path.join(THIS_DIR, 'tmp_runs', run_tag)
    train_images_dir = os.path.join(run_dir, 'train_images')
    # Eval images must be rendered per-candidate with THIS candidate's draw fn.
    eval_images_dir = os.environ.get(
        'MVC_EVAL_IMAGES_DIR',
        os.path.join(run_dir, 'eval_images'),
    )
    lora_dir = os.path.join(run_dir, 'lora')
    os.makedirs(run_dir, exist_ok=True)

    render_py = os.path.join(THIS_DIR, 'render_images.py')
    sft_train_py = os.path.join(THIS_DIR, 'sft_train.py')
    sft_eval_py = os.path.join(THIS_DIR, 'sft_eval.py')

    try:
        # --- Step 1: render training images with the candidate draw code ---
        print(f'[mvc/eval] Rendering {num_train_samples} training images...', flush=True)
        r = subprocess.run(
            [py, render_py,
             '--code_path', code_path,
             '--json_path', train_json,
             '--out_dir', train_images_dir,
             '--count', str(num_train_samples),
             '--workers', str(render_workers),
             '--prefix', 'train'],
            capture_output=True, text=True,
        )
        print(r.stdout)
        if r.returncode != 0 or 'RENDER_DONE' not in r.stdout:
            print(r.stderr, file=sys.stderr)
            raise RuntimeError('Rendering training images failed.')

        # --- Step 2: render the (fixed) eval images with the SAME candidate code ---
        print(f'[mvc/eval] Rendering {num_eval_samples} eval images...', flush=True)
        r = subprocess.run(
            [py, render_py,
             '--code_path', code_path,
             '--json_path', eval_json,
             '--out_dir', eval_images_dir,
             '--count', str(num_eval_samples),
             '--workers', str(render_workers),
             '--prefix', 'eval'],
            capture_output=True, text=True,
        )
        print(r.stdout)
        if r.returncode != 0 or 'RENDER_DONE' not in r.stdout:
            print(r.stderr, file=sys.stderr)
            raise RuntimeError('Rendering eval images failed.')

        # --- Step 3: SFT fine-tune the base VLM on the rendered training images ---
        print('[mvc/eval] Running SFT training...', flush=True)
        train_cmd = [
            py, sft_train_py,
            '--model_name', base_model,
            '--train_json', train_json,
            '--images_dir', train_images_dir,
            '--image_prefix', 'train',
            '--num_train_samples', str(num_train_samples),
            '--output_dir', lora_dir,
            '--code_path', code_path,
        ]
        if max_steps > 0:
            train_cmd += ['--max_steps', str(max_steps)]
        else:
            train_cmd += ['--num_train_epochs', str(num_train_epochs)]

        r = subprocess.run(train_cmd, capture_output=True, text=True)
        print(r.stdout[-8000:])
        if r.returncode != 0 or 'TRAIN_DONE' not in r.stdout:
            print(r.stderr[-8000:], file=sys.stderr)
            raise RuntimeError('SFT training failed.')

        # --- Step 4: evaluate the fine-tuned VLM on the fixed eval set ---
        print('[mvc/eval] Running evaluation...', flush=True)
        r = subprocess.run(
            [py, sft_eval_py,
             '--base_model', base_model,
             '--adapter_dir', lora_dir,
             '--eval_json', eval_json,
             '--images_dir', eval_images_dir,
             '--image_prefix', 'eval',
             '--num_samples', str(num_eval_samples),
             '--code_path', code_path],
            capture_output=True, text=True,
        )
        print(r.stdout)
        if r.returncode != 0 or 'EVAL_SCORE' not in r.stdout:
            print(r.stderr, file=sys.stderr)
            raise RuntimeError('Evaluation failed.')

        score = None
        for line in r.stdout.splitlines():
            if line.startswith('EVAL_SCORE:'):
                score = float(line.split('EVAL_SCORE:')[1].strip())
        if score is None:
            raise RuntimeError('Could not parse EVAL_SCORE from evaluation output.')

        # Update run_tag to include fitness score
        timestamp = datetime.datetime.now().strftime('%m%d%H%M')
        run_tag_with_score = f"{timestamp}_ID_{os.getpid()}_{score:.4f}"

        # Rename tmp_runs directory to include score
        run_dir_with_score = os.path.join(THIS_DIR, 'tmp_runs', run_tag_with_score)
        run_dir_old = run_dir
        if os.path.exists(run_dir):
            shutil.move(run_dir, run_dir_with_score)
            run_dir = run_dir_with_score
            train_images_dir = os.path.join(run_dir, 'train_images')
            lora_dir = os.path.join(run_dir, 'lora')
            if eval_images_dir.startswith(run_dir_old):
                eval_images_dir = os.path.join(run_dir, 'eval_images')

        # Save sample images for inspection
        archive_dir = os.path.join(THIS_DIR, 'saved_images', run_tag_with_score)
        os.makedirs(archive_dir, exist_ok=True)
        # Copy first 20 training images
        for i in range(min(20, num_train_samples)):
            src = os.path.join(train_images_dir, f'train_{i:06d}_*.png')
            import glob
            matches = glob.glob(src)
            if matches:
                shutil.copy2(matches[0], archive_dir)

        # Save the candidate code that generated these images alongside them
        shutil.copy2(code_path, os.path.join(archive_dir, 'gpt.py'))
        prompt_path = os.path.join(THIS_DIR, 'prompt.py')
        if os.path.exists(prompt_path):
            shutil.copy2(prompt_path, os.path.join(archive_dir, 'prompt.py'))

        # Save LoRA model before cleanup
        if os.path.exists(lora_dir):
            lora_save_dir = os.path.join(THIS_DIR, 'saved_models', run_tag_with_score)
            os.makedirs(lora_save_dir, exist_ok=True)
            shutil.copytree(lora_dir, lora_save_dir, dirs_exist_ok=True)
            print(f'[mvc/eval] Saved LoRA model to {lora_save_dir}', file=sys.stderr, flush=True)

        print(f'[mvc/eval] Saved sample images to {archive_dir}', file=sys.stderr, flush=True)

        # Final line: bare numeric objective value (hercules.py reads this).
        # IMPORTANT: This must be the last stdout line!
        print(score)

    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
