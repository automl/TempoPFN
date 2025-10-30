import logging
import os
import random
from typing import List, Optional

import pyarrow.feather as feather
import torch

logger = logging.getLogger(__name__)


class CyclicalBatchDataset:
    """
    Dataset class that loads saved batches from continuous generation script.
    Maintains a pointer and provides cyclical access to individual samples.
    Includes enhanced logging to track data shard cycling during training.
    Supports per-rank file sharding for large-scale distributed training.
    """

    def __init__(
        self,
        batches_dir: str,
        generator_type: str,
        device: Optional[torch.device] = None,
        prefetch_next: bool = True,
        prefetch_threshold: int = 32,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the cyclical batch dataset.

        Args:
            batches_dir: Directory containing the batch arrow files
            generator_type: Type of generator (for logging)
            device: Device to load tensors to
            prefetch_next: Whether to prefetch the next batch
            prefetch_threshold: Number of remaining samples to trigger prefetching
            rank: Rank of the current process (for file sharding)
            world_size: Total number of processes (for file sharding)
        """
        self.batches_dir = batches_dir
        self.generator_type = generator_type
        self.device = device
        self.prefetch_next = prefetch_next
        self.prefetch_threshold = prefetch_threshold
        self.rank = rank
        self.world_size = world_size

        self.batch_files = self._find_batch_files()
        if not self.batch_files:
            raise ValueError(f"No batch files found in {batches_dir}")

        # --- State tracking ---
        self.current_batch_idx = 0
        self.current_sample_idx = 0
        self.current_batch_data = None
        self.next_batch_data = None
        self.prefetching_in_progress = False

        # --- NEW: Logging and cycle tracking ---
        self.visited_batch_indices = set()
        self.full_cycles_completed = 0

        # Load first batch and update tracking
        self._load_current_batch()
        self.visited_batch_indices.add(self.current_batch_idx)

        logger.info(
            f"Initialized '{self.generator_type}' dataset with {len(self.batch_files)} batches. "
            f"Current batch file: '{os.path.basename(self.batch_files[self.current_batch_idx])}' "
            f"has {len(self.current_batch_data)} samples."
        )

    def _find_batch_files(self) -> List[str]:
        """
        Find and sort batch files with per-rank sharding for distributed training.

        Each rank gets a disjoint subset of files to minimize I/O contention
        when scaling to hundreds of GPUs.
        """
        import glob

        pattern = os.path.join(self.batches_dir, "batch_*.arrow")
        all_files = sorted(glob.glob(pattern))  # Sort for deterministic sharding

        if not all_files:
            return []

        # Shard files across ranks: each rank gets every world_size-th file
        # Example with 4 ranks: rank0=[0,4,8,...], rank1=[1,5,9,...], etc.
        rank_files = [
            f for i, f in enumerate(all_files) if i % self.world_size == self.rank
        ]

        # Shuffle only within this rank's shard for variety
        random.shuffle(rank_files)

        logger.info(
            f"[Rank {self.rank}] '{self.generator_type}': Sharded {len(all_files)} files → "
            f"{len(rank_files)} files for this rank ({len(rank_files) / len(all_files) * 100:.1f}%)"
        )

        return rank_files

    def _load_batch_from_file(self, batch_file: str) -> List[dict]:
        """Load a batch from arrow file."""
        try:
            table = feather.read_table(batch_file)
            has_num_channels = "num_channels" in table.column_names
            batch_data = []
            for i in range(len(table)):
                row = {
                    "series_id": table["series_id"][i].as_py(),
                    "values": table["values"][i].as_py(),
                    "length": table["length"][i].as_py(),
                    "generator_type": table["generator_type"][i].as_py(),
                    "start": table["start"][i].as_py(),
                    "frequency": table["frequency"][i].as_py(),
                    "generation_timestamp": table["generation_timestamp"][i].as_py(),
                }
                if has_num_channels:
                    row["num_channels"] = table["num_channels"][i].as_py()
                else:
                    row["num_channels"] = 1
                batch_data.append(row)
            return batch_data
        except Exception as e:
            logger.error(f"Error loading batch from {batch_file}: {e}")
            raise

    def _load_current_batch(self):
        """Load the current batch into memory."""
        if hasattr(self, "current_batch_data") and self.current_batch_data is not None:
            del self.current_batch_data
        batch_file = self.batch_files[self.current_batch_idx]
        self.current_batch_data = self._load_batch_from_file(batch_file)
        self.current_sample_idx = 0
        logger.debug(
            f"Loaded batch {self.current_batch_idx} for {self.generator_type} "
            f"with {len(self.current_batch_data)} samples"
        )

    def _trigger_smart_prefetch(self):
        """Trigger prefetching when batch is almost exhausted."""
        if not self.prefetch_next or len(self.batch_files) <= 1:
            return
        remaining_samples = self.get_remaining_samples_in_current_batch()
        should_prefetch = (
            remaining_samples <= self.prefetch_threshold
            and self.next_batch_data is None
            and not self.prefetching_in_progress
        )
        if should_prefetch:
            self._prefetch_next_batch()

    def _prefetch_next_batch(self):
        """Prefetch the next batch."""
        if self.prefetching_in_progress:
            return
        self.prefetching_in_progress = True
        next_batch_idx = (self.current_batch_idx + 1) % len(self.batch_files)
        next_batch_file = self.batch_files[next_batch_idx]
        try:
            self.next_batch_data = self._load_batch_from_file(next_batch_file)
            logger.debug(
                f"Prefetched next batch {next_batch_idx} for {self.generator_type}"
            )
        except Exception as e:
            logger.warning(f"Failed to prefetch batch {next_batch_idx}: {e}")
            self.next_batch_data = None
        finally:
            self.prefetching_in_progress = False

    def _advance_to_next_batch(self):
        """Advance to the next batch and log the transition."""
        if hasattr(self, "current_batch_data") and self.current_batch_data is not None:
            del self.current_batch_data

        previous_batch_idx = self.current_batch_idx
        self.current_batch_idx = (self.current_batch_idx + 1) % len(self.batch_files)

        if hasattr(self, "next_batch_data") and self.next_batch_data is not None:
            self.current_batch_data = self.next_batch_data
            self.next_batch_data = None
        else:
            self._load_current_batch()

        self.current_sample_idx = 0
        self.prefetching_in_progress = False

        # --- NEW: Enhanced Logging Logic ---
        self.visited_batch_indices.add(self.current_batch_idx)

        # Calculate progress
        total_files = len(self.batch_files)
        visited_count = len(self.visited_batch_indices)
        progress_percent = (visited_count / total_files) * 100

        # Log the shard cycle event
        logger.info(
            f"\nDATA SHARD CYCLED for '{self.generator_type}': "
            f"Moved from file index {previous_batch_idx} to {self.current_batch_idx}. "
            f"Unique files visited: {visited_count}/{total_files} ({progress_percent:.1f}%)."
        )

        # Check if a full cycle has been completed
        if visited_count == total_files:
            self.full_cycles_completed += 1
            logger.info(
                f"🎉 FULL CYCLE #{self.full_cycles_completed} COMPLETED for '{self.generator_type}'! "
                f"All {total_files} data files have been visited at least once. "
                "Resetting visited set to track the next cycle."
            )
            # Reset for the next cycle count
            self.visited_batch_indices.clear()
            self.visited_batch_indices.add(self.current_batch_idx)

    def get_sample(self) -> dict:
        """Get the current sample and advance pointer."""
        if not hasattr(self, "current_batch_data") or self.current_batch_data is None:
            self._load_current_batch()
        if self.current_batch_data is None:
            raise RuntimeError("No batch data loaded")
        if self.current_sample_idx >= len(self.current_batch_data):
            self._advance_to_next_batch()
        self._trigger_smart_prefetch()
        sample = self.current_batch_data[self.current_sample_idx]
        self.current_sample_idx += 1
        return sample

    def get_samples(self, num_samples: int) -> List[dict]:
        """Get multiple samples."""
        samples = []
        for _ in range(num_samples):
            samples.append(self.get_sample())
        return samples

    def get_total_samples_in_current_batch(self) -> int:
        """Get total samples in current batch."""
        if not hasattr(self, "current_batch_data") or self.current_batch_data is None:
            return 0
        return len(self.current_batch_data)

    def get_remaining_samples_in_current_batch(self) -> int:
        """Get remaining samples in current batch."""
        if not hasattr(self, "current_batch_data") or self.current_batch_data is None:
            return 0
        return max(0, len(self.current_batch_data) - self.current_sample_idx)

    def get_info(self) -> dict:
        """Get extended dataset info, including cycle progress."""
        total_files = len(self.batch_files)
        visited_count = len(self.visited_batch_indices)
        return {
            "generator_type": self.generator_type,
            "total_batch_files": total_files,
            "current_batch_idx": self.current_batch_idx,
            "current_sample_idx": self.current_sample_idx,
            "current_batch_size": self.get_total_samples_in_current_batch(),
            "remaining_in_batch": self.get_remaining_samples_in_current_batch(),
            "unique_files_visited": visited_count,
            "cycle_progress_percent": (visited_count / total_files) * 100
            if total_files > 0
            else 0,
            "full_cycles_completed": self.full_cycles_completed,
        }