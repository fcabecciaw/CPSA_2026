"""Shared pytest configuration.

Routine tests are hermetic: they need neither hardware nor the aarch64 native
libraries. Tests marked ``target`` run only on the KV260 (``pytest -m target``).
"""
