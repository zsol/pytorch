import functools
import torch

from .allowed_functions import is_in_graph_function

from .utils import hashable

from .variables import TorchCtxManagerClassVariable, TorchInGraphFunctionVariable


"""
Map of torch objects to their tracing rules (Dynamo variables).
* TorchVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* TorchCtxManagerClassVariable: The context manager classes are supported by Dynamo. E.g., torch.no_grad
* SkipFilesVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

We explicitly list torch objects which should be wrapped as TorchCtxManagerClassVariable.
The initial list comes from the heuristic in test/dynamo/test_trace_rules.py:generate_allow_list.

For developers: If you add/remove a torch level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_torch_name_rule_map. To fix the failures:
If you are adding a new torch level API or Dynamo implementation:
* Add the name with TorchCtxManagerClassVariable to this map
  if you are adding Dynamo implementation for that context manager.
* Remove the object name from test/dynamo/test_trace_rules.ignored_torch_name_rule_set if it's there.

If you are removing an existing torch level API:
* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_torch_name_rule_set
  depends on where it is.

TODO: Add torch object names mapping to TorchVariable for in graph and constant fold functions.
TODO: We would consolidate the skipfiles.check rules into trace_rules.lookup later.
TODO: We would support explictly list objects treated as skip/inline after the skipfiles.check
and trace_rules.lookup consolidation is done. Then the explicit listing of skip/inline objects have
a higher priority, which can be used to override the skipfiles.check rules in some cases.
"""
torch_name_rule_map = {
    "torch._C.DisableTorchFunctionSubclass": TorchCtxManagerClassVariable,
    "torch.amp.autocast_mode.autocast": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.enable_grad": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.inference_mode": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.no_grad": TorchCtxManagerClassVariable,
    "torch.autograd.grad_mode.set_grad_enabled": TorchCtxManagerClassVariable,
    "torch.autograd.profiler.profile": TorchCtxManagerClassVariable,
    "torch.autograd.profiler.record_function": TorchCtxManagerClassVariable,
    "torch.cpu.amp.autocast_mode.autocast": TorchCtxManagerClassVariable,
    "torch.cuda.amp.autocast_mode.autocast": TorchCtxManagerClassVariable,
    "torch.profiler.profiler.profile": TorchCtxManagerClassVariable,
    "torch.default_generator.get_state": TorchInGraphFunctionVariable,
    "torch._C.Generator.get_state": TorchInGraphFunctionVariable,
    "torch.default_generator.set_state": TorchInGraphFunctionVariable,
    "torch._C.Generator.set_state": TorchInGraphFunctionVariable,
    "torch.onnx.is_in_onnx_export": TorchInGraphFunctionVariable,
    "torch.onnx.operators.shape_as_tensor": TorchInGraphFunctionVariable,
    "torch.overrides.is_tensor_like": TorchInGraphFunctionVariable,
    "torch.jit.is_scripting": TorchInGraphFunctionVariable,
    "torch.jit.is_tracing": TorchInGraphFunctionVariable,
}


@functools.lru_cache(None)
def get_torch_obj_rule_map():
    d = dict()
    for k, v in torch_name_rule_map.items():
        obj = load_object(k)
        assert obj not in d
        d[obj] = v
    return d


def load_object(name):
    return eval(name)


def lookup(obj):
    if not hashable(obj):
        return None
    rule = get_torch_obj_rule_map().get(obj, None)
    if rule is None:
        if is_in_graph_function(obj):
            rule = TorchInGraphFunctionVariable
        else:
            rule = None
    return rule
