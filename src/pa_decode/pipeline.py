"""Bring an Experiment up to a given stage. Every stage is cached, so calling this again only loads results."""
from .data import build_units
from .decode import decode_all, run_solver_checks
from .selection import point_estimates
from .train import train_all

STAGES = ['units', 'train', 'decode', 'point']


def run_until(exp, stage, verbose=True, checks=True):
    upto = STAGES.index(stage)
    build_units(exp, verbose=verbose)
    if upto >= 1:
        train_all(exp, verbose=verbose)
    if upto >= 2:
        if checks:
            run_solver_checks(exp)
        decode_all(exp, verbose=verbose)
    if upto >= 3:
        point_estimates(exp, check=checks, verbose=verbose)
    return exp
