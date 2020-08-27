from sacred import Ingredient
from src.datasets.loader import DatasetFolder
from src.datasets.sampler import CategoriesSampler
from src.datasets.transform import with_augment, without_augment
from torch.utils.data import DataLoader


dataset_ingredient = Ingredient('dataset')
@dataset_ingredient.config
def config():
    batch_size = 256
    enlarge = True
    num_workers = 4
    disable_random_resize = False
    jitter = False
    path = 'data'
    split_dir = None


@dataset_ingredient.capture
def get_dataloader(split, enlarge, num_workers, batch_size, disable_random_resize,
                   path, split_dir, jitter, aug=False, shuffle=True, out_name=False,
                   sample=None):
    # sample: iter, way, shot, query
    if aug:
        transform = with_augment(84, disable_random_resize=disable_random_resize,
                                 jitter=jitter)
    else:
        transform = without_augment(84, enlarge=enlarge)
    sets = DatasetFolder(path, split_dir, split, transform, out_name=out_name)
    if sample is not None:
        sampler = CategoriesSampler(sets.labels, *sample)
        loader = DataLoader(sets, batch_sampler=sampler,
                            num_workers=num_workers, pin_memory=False)
    else:
        loader = DataLoader(sets, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=False)
    return loader