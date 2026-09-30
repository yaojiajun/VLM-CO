import logging
import re
import inspect
import hydra
from utils.llm_client.openai import OpenAIClient
from utils.llm_client.llama_api import LlamaAPIClient
from utils.llm_client.base import BaseClient
# utils/diversity_output.py

import difflib
import logging
from typing import List, Tuple

def init_client(cfg):
    global client
    if cfg.get("model", None): # for compatibility
        model: str = cfg.get("model")
        temperature: float = cfg.get("temperature", 1.0)
        if model.startswith("gpt"):
            from utils.llm_client.openai import OpenAIClient
            client = OpenAIClient(model, temperature)
        else: # fall back to Llama API
            from utils.llm_client.llama_api import LlamaAPIClient
            client = LlamaAPIClient(model, temperature)
    else:
        client = hydra.utils.instantiate(cfg.llm_client)
    return client



def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if  len(log) > 0:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name


def build_llm_client(model_name: str, cfg) -> BaseClient:
    from utils.llm_client.openai import OpenAIClient
    temperature = cfg.llm_client.get("temperature", 1.0)
    api_key = cfg.llm_client.get("api_key", None)
    return OpenAIClient(model=model_name, temperature=temperature, api_key=api_key)


def compute_output_similarity(output1: str, output2: str) -> float:
    """
    使用 difflib 计算两个输出字符串的相似度，范围 [0, 1]
    """
    return difflib.SequenceMatcher(None, output1.strip(), output2.strip()).ratio()


def analyze_population_output_similarity(
    population: List[dict], threshold: float = 0.9
) -> List[Tuple[int, int, float]]:
    """
    找出种群中输出相似度高于阈值的个体对。
    Returns: list of (idx1, idx2, similarity_score)
    """
    results = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            out1 = population[i].get("stdout", "")
            out2 = population[j].get("stdout", "")
            if not out1 or not out2:
                continue
            sim = compute_output_similarity(out1, out2)
            if sim >= threshold:
                results.append((i, j, sim))
    return results


def detect_low_output_diversity(
    population: List[dict], threshold: float = 0.9, ratio_limit: float = 0.7
) -> bool:
    """
    如果高相似输出的个体对超过给定比例，则返回 True 表示输出缺乏多样性。
    """
    similar_count = 0
    total_pairs = 0

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            out1 = population[i].get("stdout", "")
            out2 = population[j].get("stdout", "")
            if not out1 or not out2:
                continue
            sim = compute_output_similarity(out1, out2)
            if sim >= threshold:
                similar_count += 1
            total_pairs += 1

    if total_pairs == 0:
        return False

    ratio = similar_count / total_pairs
    logging.info(f"[Output Diversity] Similar output ratio: {ratio:.2f}")
    return ratio > ratio_limit
