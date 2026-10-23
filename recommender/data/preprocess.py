from typing import Any, Dict, List, Optional, Tuple, TypeAlias, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

# Type aliases for clarity
UserId: TypeAlias = str  # User IDs are SSO IDs (varchar strings)
ItemId: TypeAlias = Any


def construct_interaction_matrix(
    interactions: pl.DataFrame,
    dtype: Union[np.dtype[np.floating[Any]], type[np.floating[Any]]] = np.float32,
) -> Tuple[
    csr_matrix,
    List[UserId],
    List[ItemId],
    Dict[UserId, int],
    Dict[ItemId, int],
]:
    """
    Build a user x item CSR interaction matrix from a Polars DataFrame.

    Parameters
    ----------
    interactions : pl.DataFrame
        Polars DataFrame containing columns 'user_id' and 'item_id'.
    dtype : Union[np.dtype[np.floating], type[np.floating]], optional
        Data type for the matrix values (e.g., np.float32 or np.dtype('float32')),
        by default np.float32.

    Returns
    -------
    Tuple[
        csr_matrix,
        List[UserId],
        List[ItemId],
        Dict[UserId, int],
        Dict[ItemId, int],
    ]
        - CSR matrix of shape (num_users, num_items)
        - List of unique user IDs
        - List of unique item IDs
        - user_id → user_index mapping
        - item_id → item_index mapping
    """
    u_col, i_col = "user_id", "item_id"

    # Extract columns as numpy arrays
    users_arr: NDArray[Any] = interactions.select(u_col).to_series().to_numpy()
    items_arr: NDArray[Any] = interactions.select(i_col).to_series().to_numpy()

    # Compute unique users/items and inverse indices (fast, C-backed)
    users, user_inverse = np.unique(users_arr, return_inverse=True)
    items, item_inverse = np.unique(items_arr, return_inverse=True)

    # Integer row/column indices
    rows: NDArray[np.int32] = user_inverse.astype(np.int32)
    cols: NDArray[np.int32] = item_inverse.astype(np.int32)

    # Implicit interaction values
    vals: NDArray[np.floating[Any]] = np.ones(len(rows), dtype=dtype)

    # Build sparse matrix
    mat: csr_matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(users), len(items)),
        dtype=dtype,
    )

    # Mappings and lists
    user2idx: Dict[UserId, int] = {u: i for i, u in enumerate(users)}
    item2idx: Dict[ItemId, int] = {it: i for i, it in enumerate(items)}

    return mat, users.tolist(), items.tolist(), user2idx, item2idx


def filter_interactions(
    interactions: pl.DataFrame,
    min_user_pv: Optional[int] = None,
    max_user_pv: Optional[int] = None,
    min_item_pv: Optional[int] = None,
    max_item_pv: Optional[int] = None,
) -> pl.DataFrame:
    """
    Filter interactions based on user and item pageview thresholds.

    Parameters
    ----------
    interactions : pl.DataFrame
        Polars DataFrame containing columns 'user_id' and 'item_id'.
    min_user_pv : Optional[int], optional
        Minimum number of pageviews required for a user to be included, by default None.
    max_user_pv : Optional[int], optional
        Maximum number of pageviews allowed for a user to be included, by default None.
    min_item_pv : Optional[int], optional
        Minimum number of pageviews required for an item to be included, by default None.
    max_item_pv : Optional[int], optional
        Maximum number of pageviews allowed for an item to be included, by default None.

    Returns
    -------
    pl.DataFrame
        Filtered Polars DataFrame containing interactions that meet the specified thresholds.
    """
    # filter users and items based on thresholds
    user_interactions = interactions.group_by("user_id").len()
    item_interactions = interactions.group_by("item_id").len()

    if min_user_pv is not None:
        user_interactions = user_interactions.filter(pl.col("len") >= min_user_pv)
    if max_user_pv is not None:
        user_interactions = user_interactions.filter(pl.col("len") <= max_user_pv)

    # Filter items based on thresholds
    if min_item_pv is not None:
        item_interactions = item_interactions.filter(pl.col("len") >= min_item_pv)
    if max_item_pv is not None:
        item_interactions = item_interactions.filter(pl.col("len") <= max_item_pv)

    # Get the filtered user and item IDs
    filtered_user_ids = user_interactions["user_id"].to_list()
    filtered_item_ids = item_interactions["item_id"].to_list()

    # Filter the interactions dataframe
    filtered_interactions = interactions.filter(
        (pl.col("user_id").is_in(filtered_user_ids))
        & (pl.col("item_id").is_in(filtered_item_ids))
    )
    return filtered_interactions


def construct_interaction_matrix_deterministic(
    interactions: pl.DataFrame, dtype=np.float32
) -> Tuple[csr_matrix, List[Any], List[Any], Dict[Any, int], Dict[Any, int]]:
    """
    Build a user x item CSR interaction matrix from a Polars DataFrame.
    Accepts DataFrame with columns 'user_id' and 'item_id'.

    Deterministic: users and items are sorted deterministically so that
    repeated calls with the same data always produce identical outputs.
    """
    u_col, i_col = "user_id", "item_id"

    def _deterministic_sorted_unique(series: pl.Series) -> List[Any]:
        # get python list of values
        vals = series.to_list()
        # remove duplicates (order irrelevant because we'll sort)
        uniq = set(vals)
        try:
            # try natural sorting (works for homogeneous comparable types)
            return sorted(uniq)
        except TypeError:
            # fallback: sort by string representation (deterministic)
            return sorted(uniq, key=lambda x: str(x))

    # get unique users and items deterministically
    users: List[Any] = _deterministic_sorted_unique(
        interactions.select(u_col).to_series()
    )
    items: List[Any] = _deterministic_sorted_unique(
        interactions.select(i_col).to_series()
    )

    user2idx: Dict[Any, int] = {u: idx for idx, u in enumerate(users)}
    item2idx: Dict[Any, int] = {it: idx for idx, it in enumerate(items)}

    # map each row's user/item to indices (preserve row order of interactions)
    user_list = interactions.select(u_col).to_series().to_list()
    item_list = interactions.select(i_col).to_series().to_list()

    # create numpy arrays of row/col indices
    rows = np.array([user2idx[u] for u in user_list], dtype=np.int32)
    cols = np.array([item2idx[it] for it in item_list], dtype=np.int32)

    # values: implicit interactions (1 per pageview). Use provided dtype.
    vals = np.ones(len(rows), dtype=dtype)

    mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)), dtype=dtype)

    return mat, users, items, user2idx, item2idx


def construct_interaction_matrix_legacy(interactions: pl.DataFrame):
    """
    Interaction matrix ported from old recommender module.
    """
    data = interactions.to_pandas()
    users = data.user_id.unique()
    items = data.item_id.unique()
    u2i = {u: i for i, u in enumerate(users)}
    i2i = {it: i for i, it in enumerate(items)}
    rows = data.user_id.map(u2i).values
    cols = data.item_id.map(i2i).values
    vals = np.ones(len(data), dtype=np.int8)
    mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(items)))
    return mat, users, items, u2i, i2i
