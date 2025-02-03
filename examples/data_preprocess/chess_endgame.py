import os
import argparse
import chess
import chess.engine

# Load lichess_db_puzzle.csv into memory in a way that is easy to search and filter
import pandas as pd
from datasets import Dataset


def make_prefix(dp, template_type="base"):
    puzzle = dp['puzzle']
    solution = dp['solution']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: You are a chess grandmaster. For this board configuration in FEN: {puzzle}, calculate the next move that will lead to checkmate, and tell me the move in UCI. Show your work in <think> </think> tags. Show your final answer in <answer> </answer> tag, for example <answer>b3b4</answer>.
Assistant: Let me solve this step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a chess grandmaster. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n For this board configuration in FEN: {puzzle}, calculate the next move that will lead to checkmate, and tell me the move in UCI. Show your work in <think> </think> tags. Show your final answer in <answer> </answer> tag, for example <answer>b3b4</answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
    return prefix

# Example function to search and filter the DataFrame
def search_mates(df, themes=None, min_move_count=4, max_move_count=8):
    filtered_df = df
    
    if themes is not None:
        filtered_df = filtered_df[filtered_df['Themes'].apply(lambda x: any(theme in x.lower() for theme in themes))]

    filtered_df = filtered_df[filtered_df['Moves'].apply(lambda x: len(x.split(' ')) >= min_move_count and len(x.split(' ')) <= max_move_count)]

    return filtered_df

def make_puzzles(count=500):
    # Load the CSV file into a DataFrame
    # Get the file from https://database.lichess.org/lichess_db_puzzle.csv.zst
    df = pd.read_csv('lichess_db_puzzle.csv')

    filtered_puzzles = search_mates(df, themes=['mate'], min_move_count=4, max_move_count=8)

    samples = {
        "puzzle": [],
        "solution": []
    }

    # Loop through each of the puzzles and print the FEN string
    for index, row in filtered_puzzles.iterrows():
        # Get FEN and Moves column from the row

        fen = row['FEN']
        moves = row['Moves'].split(" ")

        # Initialize chess board with FEN string
        board = chess.Board(fen)

        # Apply the first move
        move = chess.Move.from_uci(moves[0])
        board.push(move)

        # Print out the board FEN after the first move
        puzzle = board.fen()
        solution = moves[1]

        # Save puzzle, solution as a row in CSV
        samples["puzzle"].append(puzzle)
        samples["solution"].append(solution)

    dataset = Dataset.from_dict(samples)
    return dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/chess')
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    raw_dataset = make_puzzles()

    print(f"Number of puzzles: {len(raw_dataset)}")

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, template_type="base")
            solution = {
                "puzzle": example['puzzle'],
                "solution": example['solution']
            }
            data = {
                "data_source": "chess_endgame",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
