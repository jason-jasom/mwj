import os
import logging
import pandas as pd
from framework import Library, StrategyExtractor, ColorFormatter
from commons import parse_json_str
import argparse
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR")


def config():
    config = argparse.ArgumentParser()
    config.add_argument("--attacker_model_name", type=str, default="mistral", choices=["mistral", "qwen2.5_instruct"])
    config.add_argument("--attacker_max_new_tokens", type=int, default=2000)
    config.add_argument("--library_name", type=str, default="strategy_library_initialized")

    return config


def construct_library(args):
    def extract_strategy(row, strategy_extractor: StrategyExtractor, library: Library):
        conv = []
        messages_max_len = 34
        for i in range(1, messages_max_len + 1):
            msg = row[f"message_{i}"]
            if pd.isna(msg):
                break
            msg_json = parse_json_str(msg, unicode_escape=False)
            if msg_json is None or "body" not in msg_json:
                break
            conv.append(msg_json["body"])

        new_strategy = strategy_extractor.extract_strategy(row["task"], conv, 10, library.get_all_in_txt())

        new_strategy["task"] = row["task"]
        new_strategy["example"] = conv
        new_strategy["score"] = 10

        logger.info(f"Row {row.name}")
        library.add(new_strategy, False, if_notify=True)

    harmbench_df = pd.read_csv(os.path.join(DATA_DIR, "harmbench_all", "data.csv"))
    mhj_df = pd.read_csv(os.path.join(DATA_DIR, "mhj", "data.csv"))

    harmbench_df.index = range(2, 2 + len(harmbench_df))
    mhj_df["task"] = mhj_df["question_id"].map(harmbench_df["Behavior"])
    mhj_df.rename(columns={"Behavior": "task"}, inplace=True)
    mhj_df.drop(columns=["Source", "temperature", "time_spent", "message_0"], inplace=True)
    # Drop message columns that are empty across all rows
    mhj_df.dropna(axis=1, how="all", inplace=True)

    library = Library()
    strategy_extractor = StrategyExtractor(args.attacker_model_name, args.attacker_max_new_tokens)

    logger.info(f"Total number of jailbreak conversations: {len(mhj_df)}")
    mhj_df.apply(lambda row: extract_strategy(row, strategy_extractor, library), axis=1)

    library.save(args.library_name)


def remove_overlaps(args):
    library = Library()
    library.load(args.library_name)

    harmbench_tasks = set(
        pd.read_csv(os.path.join(DATA_DIR, "harmbench", "data.csv"))["task"].drop_duplicates().tolist()
    )
    strategies_keys = list(library.strategies_dict.keys())
    for s_name in strategies_keys:
        i = 0
        while i < len(library.strategies_dict[s_name]["tasks"]):
            if library.strategies_dict[s_name]["tasks"][i] in harmbench_tasks:
                logger.info(f"From [{s_name}] removing tasks [{library.strategies_dict[s_name]['tasks'][i]}]")
                library.strategies_dict[s_name]["tasks"].pop(i)
                library.strategies_dict[s_name]["examples"].pop(i)
                library.strategies_dict[s_name]["tasks_embeddings"].pop(i)
                library.strategies_dict[s_name]["scores"].pop(i)
            else:
                i += 1
        if len(library.strategies_dict[s_name]["tasks"]) == 0:
            logger.info(f"Strategy [{s_name}] has no tasks left, removing it from the library.")
            del library.strategies_dict[s_name]

    library.save(f"{args.library_name}_no_test_tasks")


def check_overlaps(args):
    library = Library()
    library.load(args.library_name)

    harmbench_tasks = set(
        pd.read_csv(os.path.join(DATA_DIR, "harmbench", "data.csv"))["task"].drop_duplicates().tolist()
    )

    for s_name, s_info in library.strategies_dict.items():
        for task in s_info["tasks"]:
            if task in harmbench_tasks:
                logger.info(f"Strategy [{s_name}] has task [{task}] which is in harmbench tasks.")
    logger.info("Overlap check completed.")


if __name__ == "__main__":
    args = config().parse_args()

    # Logging setup
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "library_loading.log")
    logger = logging.getLogger("CustomLogger")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = ColorFormatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Initialize the strategy library with human multi-turn attack dataset
    logger.info("START")
    # construct_library(args)

    # remove_overlaps(args)

    check_overlaps(args)
