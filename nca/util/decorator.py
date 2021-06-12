import inspect
from collections import defaultdict
from inspect import Signature


# ========================================================================= #
# Signature Merging                                                         #
# -- yes.. I know this is overkill, but I wanted an easier way to use       #
#    fire with pytorch lightning, instead of it being two steps             #
# ========================================================================= #


class SignatureError(Exception):
    def __init__(self, param, fn, sig=None):
        if sig is None:
            sig = inspect.signature(fn)
        super().__init__(f'{repr(str(param.kind))} parameter: {repr(param.name)} is not allowed! Parameter is from function: {fn} with signature: {sig}')


IGNORE = object()


def _merge_fn_signatures(*fns):
    fn_params = {}
    param_to_fn = {}
    for fn in fns:
        sig = inspect.signature(fn)
        # get function kwargs
        params_kwargs = []
        for param in sig.parameters.values():
            # only allow parameters with default values
            if param.default == param.empty:
                    raise SignatureError(param, fn, sig)
            if param.default is IGNORE:
                continue
            # make sure the parameter with the same name does not already exist in other functions
            if param.name in param_to_fn:
                raise ValueError(f'duplicate parameter: {repr(param.name)} is not allowed, contained in both: {fn} and: {param_to_fn[param.name]}')
            param_to_fn[param.name] = fn
            # Add & modify the parameter:
            # - Signature may indicate POSITIONAL_OR_KEYWORD are out of order, we
            #   need to adjust their kind and treat them as KEYWORD_ONLY only params
            params_kwargs.append(param.replace(kind=param.KEYWORD_ONLY))
        fn_params[fn] = params_kwargs
    # create new signature
    signature = Signature([param for params in fn_params.values() for param in params])
    param_name_to_fn = {param.name: fn for fn, params in fn_params.items() for param in params}
    # return values
    return signature, param_name_to_fn


def merge_kwargs(*fns, sort_kwargs=True):
    assert fns, 'fns cannot be empty'

    def decorator(func):
        new_sig, key_to_fn = _merge_fn_signatures(*fns)

        # sort the signature alphabetically if needed

        func_sig = inspect.signature(func)
        # check the signature against the number of functions -- TODO: lift this limitation
        if len(func_sig.parameters) != len(fns):
            raise ValueError(f'parameter count miss-match: {len(func_sig.parameters)} with number of merged functions: {len(fns)}. In decorated function: {func} with signature: {func_sig} requires one parameter for each function: {fns}')
        # check signature against decorated func
        shared_params = set(key_to_fn.keys()) & set(param.name for param in func_sig.parameters.values())
        if shared_params:
            raise ValueError(f'conflicting parameter: {shared_params} with merged functions! In decorated function: {func} with signature: {func_sig}')

        def new_fn(*args, **kwargs):
            # generate kwargs
            leftover_kwargs, fn_kwargs = {}, defaultdict(dict)
            for k, v in kwargs.items():
                if k in key_to_fn:
                    fn_kwargs[key_to_fn[k]][k] = v
                else:
                    leftover_kwargs[k] = v
            # make sure we do not have extra args or kwargs -- TODO: lift this limitation
            if args or kwargs:
                raise RuntimeError('extra args and kwargs not found in the merged function signature were passed!')
            # pass all the kwargs to the function
            return func(*(fn_kwargs[fn] for fn in fns), *args, **leftover_kwargs)

        # adjust signature
        new_fn.__signature__ = new_sig

        return new_fn
    return decorator


def merge_fns(*fns):
    @merge_kwargs(*fns)
    def new_fn(*fn_kwargs):
        assert len(fn_kwargs) == len(fns)
        return tuple(fn(**kwargs) for kwargs, fn in zip(fn_kwargs, fns))
    return new_fn


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

