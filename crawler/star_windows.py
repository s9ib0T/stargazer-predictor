# star bands for sharding the crawl
# github search caps at 1000 results per query, so one "stars:>=50" wont reach the small repos
# split the range into log-spaced bands instead, each band is a shard
# narrow at the bottom (tons of repos), wide at the top (few)

# log-spaced edges: 50, 100, 200, ... each one doubles. last edge starts the open-ended top band
DEFAULT_EDGES = [50 * 2**i for i in range(12)]  # 50 .. 102400


def make_bands(edges=None):
    # edges -> list of (lo, hi). inclusive lo, inclusive hi. last band hi=None (open)
    edges = edges or DEFAULT_EDGES
    bands = []
    for i in range(len(edges) - 1):
        bands.append((edges[i], edges[i + 1] - 1))
    bands.append((edges[-1], None))
    return bands


def band_query(lo, hi):
    # github search qualifier for a star band
    if hi is None:
        return f"stars:>={lo}"
    return f"stars:{lo}..{hi}"


def band_label(lo, hi):
    return f"{lo}_{hi if hi is not None else 'max'}"