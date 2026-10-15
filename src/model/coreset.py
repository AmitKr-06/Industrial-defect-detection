"""
Coreset utilities for PathCore.

Two functions:
    greedy_coreset : picks the most representative patches (used to BUILD the memory bank)
    nn_distance    : distance from query patches to their nearest bank patch (used to SCORE images)
"""
import torch
import torch.nn.functional as F

from src.logger import logging


@torch.inference_mode()
def greedy_coreset(features: torch.Tensor, n_select: int, proj_dim: int = 128,
                   seed: int = 42, device = "cuda") -> torch.Tensor:
    """
    What it does : greedy k-center selection. Picks n_select patches so thta every other
                   patch is close to a latest one chosen patch.
    What we do   : 
      1. project the D-dim patches to proj_dim (128) with a fixed random matrix and
        L2-normalise them. Distances on 128 dims are much cheaper than on 1024 dims
        and still preserve the neighbourhood structure.
      2. start from one random patch (fixed by the seed, so the result is reproducible).
      3. keep d[i] = distance from patch i to the closet chosen patch. Repeatedly choose
         the patch with the largest d[i] (the worst-covered one), then update d.
      4. chosen patches get d = -1 so they cannot be selected twice.
    why it matters : the bank is 10x smaller than the full set (83 MB instead of 830 MB) and
                     ~7x faster to search, with almost the same accuracy (Exp 8).
    Args           : features (N, D) already on 'device'; n_select = how many patches to kepp.
    Returns        : LongTensor (n_select,) of indices, in selection order, on 'device'.
                     Because of that order, the first 1% / 5% / 10% of the result is itself a 
                     valid smaller coreset.
    """
    N, D = features.shape
    assert 0 < n_select <= N, f"Cannot select {n_select} from {N}"

    # Fixed random projection (CPU generator -> same numbers on every machine)
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(D, proj_dim, generator=g).to(device)
    fp = F.normalize(features @ proj, dim=1)

    first = torch.randint(0, N, (1,), generator=g).item()
    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = first

    d = 1.0 - fp @ fp[first]    # cosine distance of every patch to the chosen set
    d[first] = -1.0

    for j in range(1, n_select):
        idx = torch.argmax(d)   # worst-covered patch
        selected[j] = idx
        d = torch.minimum(d, 1.0 - fp @ fp[idx])
        d[idx] = -1.0

    logging.debug("Greedy coreset selected %d of %d patches", n_select, N)
    return selected

@torch.inference_mode()
def nn_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """
    What it does : for every query patch, returns the Euclidean distance to it's nearest
                   patch in the memory bank. This distance IS the anomaly score of the patch:
                   small = looks like something seen in normal images, large = unusual.
    What we do   : process the queries in chunks of 512 (torch.cdist builds a chunk x bank_size
                   matrix, so chunking keeps memory small enough for a 4 GB GPU). Queries and 
                   bank must be on the same device.
    Args         : queries (Q, D), bank (M, D).
    Returns      : tensor (Q, ) of nearest-neighbour distances.
    
    """
    assert queries.shape[1] == bank.shape[1], "Query and bank dimensions differ"
    out = []
    for i in range(0, queries.shape[0], chunk):
        out.append(torch.cdist(queries[i:i + chunk], bank).min(dim=1).values)
    return torch.cat(out)
