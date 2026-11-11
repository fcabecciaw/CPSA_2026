"""Native bindings: the vendored MATLAB Coder libraries and the local C kernels.

``lib/`` holds the aarch64 MATLAB Coder libraries vendored for the classifier
(see ``lib/PROVENANCE.md``); ``libcpsa_kalman2d.so`` in the same directory is
built from ``native/kalman2d`` by the tests on the host and by
``deploy/deploy.sh`` on the board, so it is a git-ignored artifact.
"""
