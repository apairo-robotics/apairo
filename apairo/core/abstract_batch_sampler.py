from torch.utils.data import BatchSampler

from .abstract_sampler import AbstractSampler


class AbstractBatchSampler(BatchSampler):
    r"""Every :class:`BatchSampler` should inherit from this class.

    Args:
        sampler (AbstractSampler) : The sampler that will be batched
        batch_size (int) : Size of mini-batch
        drop_last (bool) : If True, the sampler will drop the last batch if it is not full
    """

    def __init__(self, sampler: AbstractSampler, drop_last: bool = False):
        super().__init__(sampler, sampler.batch_size, drop_last)
        self.sampler = sampler

    def __iter__(self):
        for batch in super().__iter__():
            yield self.sampler.get_batch(batch)
