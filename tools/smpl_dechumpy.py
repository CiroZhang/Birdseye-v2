"""Rewrite an official SMPL .pkl without its chumpy objects (chumpy does not run on numpy>=1.24 /
Python 3.13), so smplx can load it. Each chumpy array is replaced by its plain numpy value.
usage: python tools/smpl_dechumpy.py third_party/tram/data/smpl/SMPL_NEUTRAL.pkl   (in place, keeps .orig)"""
import pickle, shutil, sys

import numpy as np


class _Ch:
    """Stand-in for chumpy.Ch: keeps whatever state pickle restores; the array lives in 'x' (or 'r')."""
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"x": state})

    def value(self):
        for k in ("x", "r", "_x"):
            if k in self.__dict__:
                return np.asarray(self.__dict__[k])
        raise ValueError("no array in chumpy object: %s" % list(self.__dict__))


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Ch
        if module == "scipy.sparse.csc":  # old scipy path
            module = "scipy.sparse"
        return super().find_class(module, name)


def main(path):
    with open(path, "rb") as fh:
        d = _Unpickler(fh, encoding="latin1").load()
    clean = {k: (v.value() if isinstance(v, _Ch) else v) for k, v in d.items()}
    shutil.copy(path, path + ".orig")
    with open(path, "wb") as fh:
        pickle.dump(clean, fh, protocol=2)
    print({k: getattr(v, "shape", type(v).__name__) for k, v in clean.items()})


if __name__ == "__main__":
    main(sys.argv[1])
