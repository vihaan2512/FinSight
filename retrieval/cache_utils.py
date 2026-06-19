import time
from functools import wraps

def ttl_cache(ttl_seconds: int = 300):
    def decorator(func):
        cache = {}
        @wraps(func)
        def wrapper(*args, **kwargs):
            def make_hashable(val):
                if isinstance(val, (list, set)):
                    return tuple(make_hashable(x) for x in val)
                if isinstance(val, dict):
                    return tuple((k, make_hashable(v)) for k, v in sorted(val.items()))
                return val

            key = (tuple(make_hashable(a) for a in args), tuple((k, make_hashable(v)) for k, v in sorted(kwargs.items())))
            now = time.time()
            if key in cache:
                result, expiry = cache[key]
                if now < expiry:
                    return result
            result = func(*args, **kwargs)
            cache[key] = (result, now + ttl_seconds)
            return result
        wrapper.cache_clear = lambda: cache.clear()
        return wrapper
    return decorator