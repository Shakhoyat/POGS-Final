"""Import shims for running the OFFICIAL POGS code on this stack (8.OFF, deviations 2 and 3 of
docs/note/off_reproduction_preregistration.md). Import this module before anything from `pogs`.

- detectron2 / CenterNet / Detic are not installable here. Their modules are replaced by inert stubs so that the
  official `pogs.data.utils.detic_dataloader` module imports unchanged; its `load()` then reads the Detic cache that
  `experiments/off_repro/export.py` wrote from the simulator masks. If the cache were missing, the official
  `create()` would call a stub and fail loudly (never silently produce masks).
- cuML is replaced by scikit-learn with the official call arguments: `HDBSCAN(cluster_selection_epsilon,
  min_samples, min_cluster_size, allow_single_cluster)` and `NearestNeighbors(n_neighbors, algorithm, metric)`.
No official source file is modified.
"""
import importlib.abc
import importlib.machinery
import sys
import types
from unittest import mock

_STUB_PREFIXES = ("detectron2", "centernet", "pogs.dependencies.Detic")


class _Stub(mock.MagicMock):
    def __call__(self, *a, **k):  # predictor / config objects: any use outside the cache path must fail
        name = getattr(self, "_mock_name", None) or "stub"
        if name in ("setup_logger",):
            return None
        return super().__call__(*a, **k)


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name == _STUB_PREFIXES[2] or name.startswith(_STUB_PREFIXES[2] + ".") or \
                name.split(".")[0] in _STUB_PREFIXES[:2]:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []

        def __getattr__(attr, _n=spec.name):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _Stub(name=f"{_n}.{attr}")
        m.__getattr__ = __getattr__
        return m

    def exec_module(self, module):
        return None


def _install_cuml():
    from sklearn.cluster import HDBSCAN as _SkHDBSCAN
    from sklearn.neighbors import NearestNeighbors

    class HDBSCAN(_SkHDBSCAN):
        """cuML signature -> scikit-learn (same parameters, CPU)."""
        def __init__(self, cluster_selection_epsilon=0.0, min_samples=None, min_cluster_size=5,
                     allow_single_cluster=False, **kw):
            super().__init__(cluster_selection_epsilon=cluster_selection_epsilon, min_samples=min_samples,
                             min_cluster_size=min_cluster_size, allow_single_cluster=allow_single_cluster)

    cuml = types.ModuleType("cuml")
    cluster = types.ModuleType("cuml.cluster")
    hdb = types.ModuleType("cuml.cluster.hdbscan")
    neighbors = types.ModuleType("cuml.neighbors")
    hdb.HDBSCAN = HDBSCAN
    neighbors.NearestNeighbors = NearestNeighbors
    cuml.cluster, cluster.hdbscan, cuml.neighbors = cluster, hdb, neighbors
    sys.modules.update({"cuml": cuml, "cuml.cluster": cluster, "cuml.cluster.hdbscan": hdb,
                        "cuml.neighbors": neighbors})


def install():
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())
    if "cuml" not in sys.modules:
        _install_cuml()


install()
