"""Source figure allowance minus local rejection boundary; an upper bound, not realized saving."""
allowance = 4096
rejection_boundary = 96
upper_bound_avoided_steps = allowance - rejection_boundary
assert upper_bound_avoided_steps == 4000
if __name__ == '__main__':
    print(f"Up to {upper_bound_avoided_steps:,} decode steps not run, conditional on honored cancellation.")
