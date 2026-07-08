import os

# On macOS, Homebrew libomp (linked by our extension) and the OpenMP runtime
# bundled with conda/sklearn conflict when both are loaded in the same process.
# Fix: load sklearn (and therefore conda's libomp) BEFORE our extension is
# imported by any test module. Combined with KMP_DUPLICATE_LIB_OK, the two
# runtimes coexist without a crash. On Linux there is only one runtime, so
# both lines are harmless.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sklearn.cluster  # noqa: F401, E402  — must precede silhouette_approx import
