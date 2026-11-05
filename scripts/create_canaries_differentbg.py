#!/usr/bin/env python3
"""Generate Option B: a distinct background for every exposed canary copy.

Shares allocation, secrets, and row order with create_canaries_samebg.py.
Run both scripts with identical inputs, exposure levels, group size, and seed.
"""
from create_canaries_samebg import run


if __name__ == '__main__':
    run('differentbg')
