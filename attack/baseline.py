from framework import ColorFormatter, HarmbenchJudge, Judge, Target, AttackVersion
import argparse
import logging
import os

import pandas as pd
from dotenv import load_dotenv

import commons.utils


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

load_dotenv(os.path.join(ROOT_DIR, ".env"))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(SCRIPT_DIR, path))


DATA_DIR = resolve_path(os.getenv("DATA_DIR", "../data"))
RESULT_DIR = resolve_path(os.getenv("RESULT_DIR", "results"))

logger = logging.getLogger("CustomLogger")
result_logger = logging.getLogger("ResultLogger")


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model_name", type=str, default="llava")
    parser.add_argument("--target_max_new_tokens", type=int, default=300)
    parser.add_argument("--judge_model_name", type=str, default="cais_HarmBench-Mistral-7b-val-cls")
    parser.add_argument("--my_judge_model_name", type=str, default="meta-llama_Llama-3.1-8B-Instruct")
    parser.add_argument("--my_judge_max_new_tokens", type=int, default=2000)
    parser.add_argument("--num_attempts", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="advbench_tiny")
    parser.add_argument("--task_i_start_from", type=int, default=0)
    parser.add_argument("--num_tasks", type=int, default=0)
    parser.add_argument("--experiment_name", type=str, default="baseline")
    parser.add_argument("--is_adashield", action="store_true")
    parser.add_argument("--use_llama_guard", type=str, default="")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--skip_my_judge",
        action="store_true",
        help="Only run the final HarmBench judge. jailbreak_score will be left as 0.",
    )
    return parser


def run_baseline(args):
    target = Target(
        args.target_model_name,
        args.target_max_new_tokens,
        args.is_adashield,
        args.use_llama_guard,
    )
    final_judge = HarmbenchJudge(args.judge_model_name)
    my_judge = None if args.skip_my_judge else Judge(args.my_judge_model_name, args.my_judge_max_new_tokens)

    data_path = os.path.join(DATA_DIR, args.dataset, "data.csv")
    df = pd.read_csv(data_path)
    total_num_tasks = len(df)
    num_tasks = total_num_tasks - args.task_i_start_from if args.num_tasks == 0 else args.num_tasks

    results = []
    for task_i, task in enumerate(df["task"]):
        if task_i < args.task_i_start_from:
            continue

        result_logger.info(f"Task {task_i}. {task}")
        for attempt_i in range(args.num_attempts):
            target.clear_history()

            logger.info(f"{'=' * 20} Task {task_i}, Attempt {attempt_i}: {task} {'=' * 20}")
            response = target.get_response(task, None)
            logger.info(f"response: {response}")
            target.add_prompt_to_conv(task, None, AttackVersion.NO_VISION)
            target.add_response_to_conv(response)

            is_success = final_judge.eval(task, response)
            final_judge_result = "Success" if is_success else "Failure"
            logger.info(f"Final judge: {final_judge_result}")
            jailbreak_score = 0 if my_judge is None else my_judge.eval(task, response)
            response_cleaned = response.replace("\n", " ") if response else ""

            result = {
                "task_i": task_i,
                "task": task,
                "attempt_i": attempt_i,
                "is_success": is_success,
                "judge_score": int(is_success),
                "jailbreak_score": jailbreak_score,
                "rounds_num": 1,
                "steps_num": 1,
                "last_response": response_cleaned,
                "attack_version_0": target.get_attack_version(0),
                "total_target_query": target.get_total_target_query(),
            }
            results.append(result)

            result_logger.info(
                f"\tAttempt {attempt_i + 1}: Final judge: {final_judge_result}, "
                f"Jailbreak score: {jailbreak_score}, Rounds num: 1, Steps num: 1, "
                f"Total target query: {result['total_target_query']} - Last response: {response_cleaned}"
            )

        if task_i == args.task_i_start_from + num_tasks - 1 or task_i == total_num_tasks - 1:
            break

    return results


def setup_logging(result_dir, task_i_start_from):
    os.makedirs(result_dir, exist_ok=True)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(os.path.join(result_dir, f"running_{task_i_start_from}.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter("%(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    result_logger.setLevel(logging.INFO)
    result_logger.handlers.clear()

    result_file_handler = logging.FileHandler(os.path.join(result_dir, f"results_{task_i_start_from}.log"), mode="w")
    result_file_handler.setLevel(logging.INFO)
    result_file_handler.setFormatter(file_formatter)
    result_logger.addHandler(result_file_handler)


def save_results(results, result_dir, task_i_start_from):
    results_df = pd.DataFrame(results)
    binary_asr = results_df.groupby("task_i")["is_success"].any().mean()
    average_asr = results_df["is_success"].mean()
    average_rounds_num_in_success = results_df[results_df["is_success"]]["rounds_num"].mean()
    average_steps_num_in_success = results_df[results_df["is_success"]]["steps_num"].mean()
    average_success_jailbreak_score = results_df[results_df["is_success"]]["jailbreak_score"].mean()
    average_jailbreak_score = results_df["jailbreak_score"].mean()

    result_logger.info(
        f"Binary ASR: {binary_asr}; Average ASR: {average_asr}; "
        f"Average successful rounds num: {average_rounds_num_in_success}; "
        f"Average successful steps num: {average_steps_num_in_success}; "
        f"Average successful jailbreak score: {average_success_jailbreak_score}; "
        f"Average jailbreak score: {average_jailbreak_score}"
    )

    stats_path = os.path.join(result_dir, f"stats_{task_i_start_from}.csv")
    results_df.to_csv(stats_path, index=False)
    result_logger.info(f"Results saved to {stats_path}")
    return stats_path


if __name__ == "__main__":
    args = config().parse_args()
    commons.utils.PORT = args.port

    result_dir = os.path.join(RESULT_DIR, args.experiment_name)
    setup_logging(result_dir, args.task_i_start_from)

    logger.info("START BASELINE")
    baseline_results = run_baseline(args)
    save_results(baseline_results, result_dir, args.task_i_start_from)
