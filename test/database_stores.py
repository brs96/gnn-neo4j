"""Minimal in-memory fakes for DatabaseFeatureStore, DatabaseGraphStore, and
DatabaseSampler.  Used in unit tests that must run without a real database.
"""

from typing import Any

import numpy as np
from torch import Tensor
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.data.graph_store import EdgeAttr
from torch_geometric.typing import EdgeTensorType, FeatureTensorType

from gnn_neo4j.data.database_feature_store import DatabaseFeatureStore
from gnn_neo4j.data.database_graph_store import DatabaseGraphStore
from gnn_neo4j.sampler.database_sampler import DatabaseSampler


class FakeDatabaseFeatureStore(DatabaseFeatureStore):
    """In-memory DatabaseFeatureStore for unit tests.

    Seed it with numpy arrays via ``data[(group, attr_name, nid)] = row``.
    Tracks ``fetch_call_count`` so tests can assert cache bypass behaviour.
    """

    def __init__(self, data: dict[tuple, np.ndarray] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._data: dict[tuple, np.ndarray] = dict(data or {})
        self.fetch_call_count: int = 0

    def _fetch_remote_attrs(
        self,
        attr: TensorAttr,
    ) -> tuple[list[dict], list[int]]:
        self.fetch_call_count += 1
        if isinstance(attr.index, Tensor | np.ndarray):
            nids = attr.index.tolist()
        elif isinstance(attr.index, slice):
            nids = list(range(attr.index.start, attr.index.stop, attr.index.step))
        else:
            nids = [attr.index]
        key = attr.attr_name
        records = []
        fetched = []
        for nid in nids:
            row = self._data.get((attr.group_name, key, nid))
            if row is not None:
                records.append({"id": nid, key: row})
                fetched.append(nid)
        return records, fetched

    def _decode_remote_attrs(
        self,
        records: list[dict],
        attr: TensorAttr,
    ) -> np.ndarray:
        key = attr.attr_name
        if not records:
            return np.empty((0,), dtype=np.float32)
        rows = [rec[key] for rec in records]
        return np.stack(rows)

    def _put_tensor_db(self, tensor: FeatureTensorType, attr: TensorAttr) -> bool:
        if isinstance(attr.index, Tensor | np.ndarray):
            nids = attr.index.tolist()
        elif isinstance(attr.index, slice):
            nids = list(range(attr.index.start, attr.index.stop, attr.index.step))
        else:
            nids = [attr.index]
        arr = tensor.detach().cpu().numpy() if isinstance(tensor, Tensor) else tensor
        for i, nid in enumerate(nids):
            self._data[(attr.group_name, attr.attr_name, int(nid))] = arr[i]
        return True

    def _remove_tensor_db(self, attr: TensorAttr) -> bool:
        return True

    def _get_tensor_size(self, attr: TensorAttr) -> tuple[int, ...] | None:
        out = self._get_tensor(attr)
        return out.shape if out is not None else None

    def get_all_tensor_attrs(self) -> list[TensorAttr]:
        seen = set()
        out = []
        for group, name, _ in self._data:
            if (group, name) not in seen:
                seen.add((group, name))
                out.append(TensorAttr(group_name=group, attr_name=name))
        return out


class FakeDatabaseGraphStore(DatabaseGraphStore):
    """In-memory DatabaseGraphStore for unit tests.

    Configure ``records[query_key]`` to control what ``query_db``
    returns.  Tracks ``executed_queries`` list of ``(query, kwargs)`` pairs.
    """

    def __init__(self, records: dict[str, dict | None] | None = None):
        super().__init__()
        self._records: dict[str, dict | None] = dict(records or {})
        self.executed_queries: list[tuple[str, dict]] = []

    def query_db(self, query: str, params: dict) -> dict | None:
        self.executed_queries.append((query, params))
        return self._records.get(query)

    # GraphStore ABC — not exercised in sampler/gs tests but required.
    def _put_edge_index(self, edge_index: EdgeTensorType, edge_attr: EdgeAttr) -> bool:
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> EdgeTensorType | None:
        return None

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        return True

    def get_all_edge_attrs(self) -> list[EdgeAttr]:
        return []


class FakeDatabaseSampler(DatabaseSampler):
    """DatabaseSampler that issues canned queries
    against FakeDatabaseGraphStore.
    """

    def _build_node_sampling_query(self) -> str | None:
        return "FAKE_NODE_QUERY"

    def _build_edge_sampling_query(self) -> str | None:
        return "FAKE_EDGE_QUERY"

    def _build_node_query_params(self, seeds: Tensor, **kwargs: Any) -> dict:
        return {"seed_ids": seeds.tolist()}

    def _build_edge_query_params(self, seeds: Tensor, **kwargs: Any) -> dict:
        return {"seed_ids": seeds.tolist()}

    def _decode_node_sampling_record(self, record: Any, seeds: Tensor) -> None:
        return None

    def _decode_edge_sampling_record(self, record: Any, seeds: Tensor) -> None:
        return None
