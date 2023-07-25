from src.cache import cache
from src.features.aggregation import naive_all_regions
from src.features.time_series.lagged_impact import lagged_impact


@cache
def regression(max_lags=0, include_controls=True, media_source="mediacloud"):
    dfs, vars = naive_all_regions(media_source=media_source)
    if not include_controls:
        assert max_lags == 0
        lags = {(0, 0): vars.w}
        dfs = [df[vars.y + vars.w] for df in dfs]
    elif max_lags == 0:
        lags = {
            (0, 0): vars.future_only + vars.future,
        }
    else:
        lags = {
            (0, 0): vars.future_only,
            (0, max_lags): vars.future,
            (1, max_lags): vars.y,
        }
    models, results = lagged_impact(dfs, vars.y, lags)
    return results
