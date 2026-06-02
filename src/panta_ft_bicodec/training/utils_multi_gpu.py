import os

import torch.distributed as dist
import idr_torch 
import torch
import logging



def setup_distributed() -> torch.device:
    """
    Setup the distributed training environment. On Jean Zay
    """
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=idr_torch.size,
        rank=idr_torch.rank,
    )
    local_rank = idr_torch.local_rank
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def display_string_log_multi_gpu(string: str, rank: int) -> str:
    """ Useless function to make the general display more readable"""
    if rank == 0: 
        return logging.info(f"==== {string} ====")

def save_optimizers_and_steps(
        optimizers: dict[str, list[torch.optim.Optimizer]],
        schedulers: dict,
        steps: int,
        epochs_training_steps: int,
        epochs: int,
        checkpoint_path: str
    ) -> None:
    torch.save({
        'optimizer_state_dict': {task.value.name: [opt.state_dict() for opt in optimizers[task]] for task in optimizers},
        'scheduler_state_dict': {task.value.name: [scheduler.state_dict() for scheduler in schedulers[task]] for task in schedulers},
        'step': steps,
        'epochs_training_steps': epochs_training_steps,
        'epoch': epochs
    }, str(checkpoint_path))



def setup_ddp(type_ddp: str) -> None:
    match type_ddp:
        case "jeanzay":
            local_device = setup_distributed()
            rank = idr_torch.rank  # ou torch.distributed.get_rank()
            world_size = idr_torch.size
            local_rank = idr_torch.local_rank
        case "single_node":
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])  # Local GPU ID on the current machine
            world_size = int(os.environ["WORLD_SIZE"])  # Total number of GPUs (processes)
            local_device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(local_rank)
            rank = local_rank
        case _:
            raise ValueError("type_ddp should be either 'jeanzay' or 'single_node'")
    is_main_process = rank == 0
    return local_device, rank, world_size, local_rank, is_main_process



def save_optimizers_and_steps(
        optimizer_generator,
        optimizer_discriminator,
        scheduler_discriminator,
        scheduler_generator,
        epoch,
        steps,
        checkpoint_path,
    ) -> None:
    torch.save({
        'optimizer_generator': optimizer_generator.state_dict(),
        'optimizer_discriminator': optimizer_discriminator.state_dict(),
        'scheduler_discriminator': scheduler_discriminator.state_dict(),
        'scheduler_generator': scheduler_generator.state_dict(),
        'steps': steps,
        'epoch': epoch
    }, str(checkpoint_path))