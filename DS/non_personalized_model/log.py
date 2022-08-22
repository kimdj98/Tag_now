import logging

logging.basicConfig('user_data.log', level = logging.INFO, format = )

def logged(fn):
    from functools import wraps
    from datetime import datetime, timezone

    @wraps(fn)
    def inner(*args, **kwargs):
        run_dt = datetime.now(timezone.utc)
        result = fn(*args, **kwargs)
        print(f'{run_dt}: called {fn.__name__}')

    return inner
