#!/usr/bin/env python

import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Iterator
import numpy as np
import torch
import torch.utils.data
import tqdm
from collections import Counter

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)


def analyze_actions(
    dataset: LeRobotDataset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> None:
    """Analyze all actions in the dataset to find unique values."""
    
    logging.info("Loading full dataset to analyze actions")
    
    # Create dataloader for entire dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Collect all actions
    all_actions = []
    total_samples = 0
    
    logging.info("Collecting all actions...")
    
    for batch in tqdm.tqdm(dataloader, desc="Processing batches"):
        if "action" in batch:
            # Convert to numpy and round to nearest integers
            actions = batch["action"].numpy()
            # Round to handle values like 10.0 -> 10
            actions_rounded = np.round(actions).astype(int)
            all_actions.append(actions_rounded)
            total_samples += len(actions_rounded)
    
    # Concatenate all actions
    if all_actions:
        all_actions = np.concatenate(all_actions, axis=0)
        logging.info(f"Total action samples: {total_samples}")
        logging.info(f"Action shape: {all_actions.shape}")
        
        # Find unique actions
        unique_actions = np.unique(all_actions, axis=0)
        logging.info(f"Number of unique actions: {len(unique_actions)}")
        
        # Print some statistics
        print(f"\n=== ACTION ANALYSIS RESULTS ===")
        print(f"Total action samples: {total_samples}")
        print(f"Action dimensionality: {all_actions.shape[1]}D")
        print(f"Number of unique actions: {len(unique_actions)}")
        
        # Show action ranges
        min_vals = np.min(all_actions, axis=0)
        max_vals = np.max(all_actions, axis=0)
        print(f"Action ranges:")
        for dim in range(all_actions.shape[1]):
            print(f"  Dimension {dim}: [{min_vals[dim]}, {max_vals[dim]}]")
        
        # Show most common actions
        action_tuples = [tuple(action) for action in all_actions]
        action_counts = Counter(action_tuples)
        most_common = action_counts.most_common(10)
        
        print(f"\nTop 10 most frequent actions:")
        for i, (action, count) in enumerate(most_common, 1):
            percentage = (count / total_samples) * 100
            print(f"  {i:2d}. {action} - {count:6d} times ({percentage:.2f}%)")
        
        # Show unique actions (if not too many)
        if len(unique_actions) <= 50:
            print(f"\nAll unique actions:")
            for i, action in enumerate(unique_actions):
                count = action_counts[tuple(action)]
                percentage = (count / total_samples) * 100
                print(f"  {tuple(action)} - {count} times ({percentage:.2f}%)")
        else:
            print(f"\nToo many unique actions to display ({len(unique_actions)})")
            print("Showing first 20:")
            for i in range(20):
                action = unique_actions[i]
                count = action_counts[tuple(action)]
                percentage = (count / total_samples) * 100
                print(f"  {tuple(action)} - {count} times ({percentage:.2f}%)")
    
    else:
        logging.error("No actions found in the dataset!")


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    analyze_all_actions: bool = False,
) -> None:
    """Original visualization function with optional action analysis."""
    
    if analyze_all_actions:
        analyze_actions(dataset, batch_size, num_workers)
        return
    
    # Original episode-specific visualization code
    episode_sampler = EpisodeSampler(dataset, episode_index)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        sampler=episode_sampler,
    )

    logging.info("Logging episode data")

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        # iterate over the batch
        for i in range(len(batch["index"])):
            # display each dimension of action space (e.g. actuators command)
            if "action" in batch:
                for dim_idx, val in enumerate(batch["action"][i]):
                    print(f"action/{dim_idx}", val.item())

            # display each dimension of observed state space (e.g. agent position in joint space)
            if "observation.state" in batch:
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    print(f"state/{dim_idx}", val.item())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/pusht",
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to visualize (ignored if --analyze-all-actions is used).",
    )
    parser.add_argument(
        "--analyze-all-actions",
        action="store_true",
        help="Analyze all actions in the entire dataset instead of visualizing a single episode.",
    )
    parser.add_argument(
        "--local-files-only",
        type=int,
        default=0,
        help="Use local files only. By default, this script will try to fetch the dataset from the hub if it exists.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )

    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    repo_id = args.repo_id
    root = args.root
    local_files_only = bool(args.local_files_only)

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, root=root, local_files_only=local_files_only)
    
    visualize_dataset(
        dataset, 
        args.episode_index, 
        args.batch_size, 
        args.num_workers,
        args.analyze_all_actions
    )


if __name__ == "__main__":
    main()