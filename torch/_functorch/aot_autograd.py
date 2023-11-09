import collections
import dataclasses
import itertools
import logging
import warnings
import pprint
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, NewType
from unittest.mock import patch

from functorch import make_fx

import torch
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch import Tensor
from torch._subclasses.meta_utils import safe_is_leaf
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import dynamo_timed, lazy_format_graph_code, preserve_rng_state
from torch._guards import detect_fake_mode, tracing
from torch._prims_common import CUDARngStateHelper
from torch._logging import getArtifactLogger
from torch._subclasses import FakeTensor, FakeTensorMode
from torch._subclasses.fake_tensor import is_fake
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch.fx import Interpreter
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.symbolic_shapes import (
    ShapeEnv, is_concrete_int, fx_placeholder_vals, definitely_true, definitely_false, sym_eq
)
from torch.multiprocessing.reductions import StorageWeakRef
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass, transform_subclass
from torch._decomp.decompositions_for_rng import PhiloxStateTracker, rng_decompositions
from . import config
from .partitioners import default_partition
from torch._guards import TracingContext, DuplicateInputs, Source
from torch.utils.weak import WeakIdRef

original_zip = zip

def strict_zip(*iterables, strict=True, **kwargs):
    if not strict:
        return original_zip(*iterables, **kwargs)

    shortest_length = min(len(it) for it in iterables)
    for iterable in iterables:
        if len(iterable) != shortest_length:
            raise ValueError("The iterables have different lengths and strict mode is enabled.")

    return original_zip(*iterables, **kwargs)

zip = strict_zip

log = logging.getLogger(__name__)
aot_joint_log = getArtifactLogger(__name__, "aot_joint_graph")
aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")

MutationType = Enum(
    "MutationType", ("none", "metadata_only", "data", "data_and_metadata")
)
OutputType = Enum(
    "OutputType", (
        # output is not an alias
        "non_alias",
        # output aliases an input
        "alias_of_input",
        # output **is** an input tensor
        "is_input",
        # output has a ._base tensor, which is a graph intermediate.
        # We need to return its ._base as a graph output,
        # so its requires_grad info is populated correctly.
        # Instructs the runtime code to regenerate the current output
        # from a base tensor, graph_intermediates[base_idx]
        "alias_of_intermediate_save_as_output",
        # Same as above; but we don't need to explicitly add its ._base
        # as a graph output, because it already **is** a graph output.
        "alias_of_intermediate",
        # Same as above; but the output's ._base is **already** a user output.
        # Instructs the runtime code to regenerate the current output from
        # a base tensor, user_outputs[base_idx]
        "alias_of_intermediate_base_is_user_output",
        # See Note [Intermediate Bases Optimization]
        "unsafe_view_alias",
        # output is an alias, but has a custom autograd.Function backward.
        # In this case, we don't want to do view-replay, since we won't be able to replay the custom function.
        # Instead, we'll treat this output "normally", and trace its backward into the graph.
        "custom_function_view",
    )
)

def partial_asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
    elif isinstance(obj, (list, tuple)):
        return obj.__class__([partial_asdict(item) for item in obj])
    elif isinstance(obj, dict):
        return {k: partial_asdict(v) for k, v in obj.items()}
    else:
        return obj

aten = torch.ops.aten

# This global counter increments every time we compile a graph with
# AOTAutograd.  You can use this to correlate runtime error messages
# with compile time (e.g., if you get an error at runtime saying
# compiled graph 3 failed, you can set a breakpoint at compile time
# for this graph number to investigate further at compile time.)
#
# NB: this is different from get_aot_compilation_context, which tracks
# each underlying graph that is compiled.  In contrast, AOT_COUNTER
# corresponds to top-level invocations of aot_module/aot_function;
# one counter is allocated per entire compiled block (but this block
# may involve compiling multiple subgraphs; e.g., for forwards/backwards)
AOT_COUNTER = itertools.count()

KNOWN_TYPES = tuple(
    [torch.Tensor, int, str, float, bool, type(None)] + list(py_sym_types)
)

# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False

def setup_stacktrace_preservation_hooks(roots: List):
    def iter_graph(roots):
        if not roots:
            return
        seen = set()
        q = collections.deque()
        for node in roots:
            if node is not None:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_):
        def callback():
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback

    def get_prehook(stack_, seq_nr):
        def prehook(grad_output):
            global callback_set

            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(
                    get_callback(fx_traceback.format_stack())
                )
                callback_set = True

            fx_traceback.set_stack_trace(stack_)
            fx_traceback.set_grad_fn_seq_nr(seq_nr)

        return prehook

    def get_posthook(special_stack_, seq_nr):
        def posthook(grad_input, grad_output):
            fx_traceback.set_stack_trace(special_stack_)
            fx_traceback.reset_grad_fn_seq_nr()

        return posthook

    for node in iter_graph(roots):
        forward_node_stack = node.metadata.get("traceback_", [])
        node.register_prehook(get_prehook(forward_node_stack,
                              node._sequence_nr()))

        special_stack = forward_node_stack.copy()
        special_stack.append(
            "Gradient addition node due to multiple use of tensor around:"
        )
        node.register_hook(get_posthook(special_stack, node._sequence_nr()))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# AOT Autograd contains a pretty non-trivial amount of logic to handle edge cases around aliasing and mutation
# that are external to the graph (they show up as side effects in some way when you run the graph).
#
# Take a look at `test_aotdispatch.py TestAOTAutograd.test_input_mutation*` tests for some examples functions
# and what they're compiled graphs looks like.
# Below is a very long comment detailing several edge cases, and showing how AOT Autograd handles them.
#
# Note [AOT Autograd: input data mutations]
#
# If we compile a function that mutates inputs, then those input mutations are real side effects
# that a user expects to see after running the compiled graph.
# However, the graph that we want to send to a backend needs to be *entirely* functional.
# The way we reconcile this difference is that we remove the mutations completely from the graph that we compile
# but we update the graph to return (updated_inputs, user_outputs).
# In the epilogue that runs after the compiled graph is executed, we copy the updated inputs back to the originals.
#
# Example: original user code:
# def f(x):
#     x.mul_(2)
#     out = x.mul(3)
#     return out
#
# After AOT Autograd compiles, we end up with a:
# (a) compiled graph
# (b) autograd.Function.forward() method, that executes the compiled graph
# (c) wrapper function, that calls the autograd.Function.forward() and performs the epilogue
#
# The output of (a, b, c) are all written below.
#
# def compiled_forward_graph(x):
#     x_updated = x.mul(2)
#     out = x_updated.mul(3)
#     return x_updated, out
#
# # x_updated gets a gradient in the compiled backward
# def compiled_backward_graph(grad_x_updated, grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.copy_(x_updated)
#     return out
#
# Another important thing to note is that updated inputs (due to data mutations) *do* participate
# in the compiled backward graph! Since the compiled forward graph gets N extra outputs
# (due to updated inputs showing up as graph outputs),
# The compiled backward gets an additional N inputs.
# That way, during the x.copy_(x_updated) bit in the epilogue, gradients will flow from the updated input
# back to the original input.


# Note [AOT Autograd: input metadata mutations]
#
# For the same reason as input mutations, we also don't put input metadata mutations in the graph.
# Instead, we return the updated version of the input (a view), and mutate the input's metadata outside of the graph
#
# Example: original user code:
# def f(x):
#     x.t_()
#     out = x.mul(3)
#     return out
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     x_updated = x.t()
#     out = x_updated.mul(3)
#     return x_updated, out
#
# # x_updated does *not* get a gradient in the compiled backward
# def compiled_backward_graph(grad_out):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     x_updated, out = compiled_forward_graph(x)
#     return x_updated, out
#
# def compiled_wrapper(x):
#     x_updated, out = autograd.Function.apply(x)
#     x.as_strided_(x_updated)
#     return out


# Note [AOT Autograd: outputs aliasing inputs or intermediates!]
#
# AOT Autograd needs special handling for outputs that alias graph inputs or intermediates!
# Why?
# (1) autograd.Function.forward() has a limitation, where views that returned in the forward cannot later be mutated.
# (2) views don't need to be compiled in the graph anyway - it's cheap to generate them outside of the compiled graph,
#     in an epilogue.
# For outputs that alias inputs, we do the following:
# (a) *still* return the aliased output as a graph output
# (b) In the AOT Autograd wrapper/epilogue, we don't return that aliased output. Instead, we use it to regenerate the output.
#
# For outputs that alias *intermediates*, we do the following:
# (a) Return the output in the compiled forward, **and** return it's ._base (a graph intermediates) as an output in the forward
# (b) Use (output, graph_intermediate) to regenerate the alias, and return that to the user (instead of the compiled fw output).
# You might wonder why we return the aliased output directly in the graph (and making the graph compute it),
# only to not return it and instead generate a fresh alias off of the intermediate,
# instead of (say) just storing metadata about the size/stride of the output somewhere to generate the alias. There are two reasons:
# (1) Getting the actual alias tensor allows us to use view-replay to generate the alias, instead of an as_strided() call
# (2) Inductor (and other backends) are free to change the memory format of graph outputs, if it results in better performance.
#     This can result in problems if a user later tries to .view() that output expecting it to have one set of strides,
#     when it has a different set of strides.
#     By including the view op directly in the graph, inductor takes that into account when deciding what memory format
#     the graph intermediate should be.
#
# Another important thing to note is how our traced backward() graph handles aliases.
# (this applies to outputs aliasing inputs, outputs aliasing intermediates,
#  *and* updated inputs returned in the compiled forward due to metadata-only mutations).
# Any outputs that alias (either inputs or intermediates) do NOT participate in the compiled backward graph
# It would be wasteful to include them in the compiled backward(), because we regenerate them eagerly
# at the end of the forward.
#
# Example: original user code:
# def f(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     return out1, out2
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(x):
#     out1 = x.t()
#     intermediate = x.mul(2)
#     out2 = intermediate.view(-1)
#     # the compiled graph also returns the intermediate
#     return out1, out2, intermediate
#
# # intermediate gets a gradient in the compiled backward.
# # both output aliases (out1 and out2) do not.
# def compiled_backward_graph(grad_intermediate):
#     grad_x = ...
#     return grad_x
#
# def autograd.Function.forward(x):
#     out1, out2, intermediate = compiled_forward_graph(x)
#     return out1, out2, intermediate
#
# def compiled_wrapper(x):
#     out1, out2, intermediate = autograd.Function.apply(x)
#     # regenerate out1 from the input
#     out1_regenerated = out1._view_func(x)
#     # regenerate out1 from the intermediate
#     out2_regenerated = out2._view_func(intermediate)
#     return out1_regenerated, out2_regenerated


# Note [AOT Autograd: mutations to inputs that alias other inputs]
#
# Another edge case that is (only partially) handled today is when an input is mutated, but itself aliases another input.
# AOT Autograd needs to **ensure** that functionalization knows that the two inputs are aliased to each other.
# That way, when the aliased input is accessed later in the graph, functionalization knows to "update" the alias
# given the mutation that occurred.
#
# This is handled by updating the calling convention: we create a "synthetic base" that becomes a new input
# in the compiled function, and we regenerate the original (aliased) inputs directly off of the base
# inside of the compiled function.
#
# This logic is fully encapsulated in aot_wrapper_synthetic_base()
#
# Example: original user code:
# def f(x, x_view):
#     x.mul_(2)
#     out = x * x_view
#     return out
# f(x, x.view(-1))
#
# AOT Autograd output (compiled graph, autograd.Function.forward(), wrapper function):
# def compiled_forward_graph(base)
#     x = generate_x(base)
#     x_view = generate_x_view(base)
#     x_updated = x.mul(2)
#     x_view_updated = x_updated.view(-1)
#     out = x_updated * x_view_updated
#     return x_updated, out
#
# # The calling convention change from (aliases) -> (base) happens
# # *outside* of the autograd.Function.forward().
# # That means the forward() only has 1 input (base),
# # and the backward() only has 1 output (grad_base)
# def compiled_backward_graph(grad_out):
#     grad_base = ...
#     return grad_base
#
# def autograd.Function.forward(base):
#     x_updated, out = compiled_forward_graph(base)
#     return x_updated, out
#
# # The compiled wrapper is where we create synthetic bases.
# # The info on which inputs are mutated is also tracked *before* synthetic base creation.
# def compiled_wrapper(x, x_view):
#     base = merge_view_inputs(x, x_view)
#     x_updated, out = autograd.Function.apply(base)
#     # x and x_view are aliased in eager mode, so this mutation to x will automatically affect x_view.
#     x.copy_(x_updated)
#     return out


# Note [AOT Autograd: Views to avoid tangents aliasing inputs]
#
# We view every forward output when creating out tangent tensors to handle the problematic
# case in which a subclass does extra aliasing between graph outputs/inputs in a way that
# is not visible above the sublass.
#
# Ordinarily, when constructing the joint function that we want to trace in AOTAutograd,
# we're guaranteed that the tangent tensors that we pass
# into the joint are distinct tensors from the primals. This is because when
# decide which forward outputs to create tangents for, we only create tangents
# for forward outputs that are not aliases of inputs (See Note
# [AOT Autograd: outputs aliasing inputs or intermediates!]).
#
# However, when wrapper tensor subclasses enter the picture, it is possible
# to have an output of the forward that is a subclass that is not an
# input / alias of an input, but one of its inner tensors is an alias!
# NestedTensor is an example: Performing an out-of-place pointwise op on a
# NestedTensor constructs a fresh NestedTensor that holds onto the input's
# offsets tensor directly.
#
# Having tangent tensors that are the same as the (primal) forward inputs,
# can cause problems during tracing as make_fx() will specialize on our
# duplicate inputs: If we passed in the same tensor for primals_1 and
# tangents_1 during tracing, make_fx() will happily sub out all usages of
# tangents_1 with primals_1 in the graph, which is not what we want.
#
# To work around this, we view every forward output when creating out tangent
# tensors so that tangents can never be the same as forward inputs even if
# forward inputs alias forward outputs.
#
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# This class stores info about every user output.
@dataclass(frozen=True)
class OutputAliasInfo:
    # Tells us if this output is:
    # (1) a regular (non-aliased) output
    # (2) an alias of a forward input
    # (3) **is** a forward input (special case of "alias_of_input")
    # (4) an alias of an intermediate (aka an alias of an output of the inner traced forward)
    # (5) an alias of an intermediate, that explicitly requires returning the intermediate
    #     as a graph output
    # (6) an alias of an intermediate, where that intermediate is also a user output
    output_type: OutputType
    # The raw type of the output (torch.Tensor, SymInt, etc)
    raw_type: type
    # If (1) above, then
    # - base_idx is None
    # If (2) or (3) above, then
    # - Tells us that the base of this alias is user_fwd_input[base_idx]
    #   (This is an index into the inputs *before* we make synthetic bases)
    # If (4) or (5) above, then
    # - Tells us that the base of this alias is output_graph_intermediates[base_idx]
    #   here, this refers to the index of the *direct* traced
    # If (6) above, then:
    # - Tells us that the base of this alias is output_user_fwds[base_idx]
    #   here, this refers to the index of the *direct* traced
    base_idx: Optional[int]
    # If it is a Tensor, what the dynamic dims are (otherwise is None)
    dynamic_dims: Optional[Set[int]]
    # requires_grad
    requires_grad: bool


# This class tells us info about user inputs.
@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool
    mutates_data: bool
    mutates_metadata: bool
    mutations_hidden_from_autograd: bool
    # This can only happen from a call to aten.set_() on a graph input.
    mutates_storage_metadata: bool
    requires_grad: bool

    def __post_init__(self):
        if self.mutates_storage_metadata:
            # For convenience, we guarantee that this is always true.
            # In practice, If we call .set_(), then at runtime there is no neid
            # to additionally fix  up the tensor metadata, since our runtime
            # call to inp.set_(updated_inp) will already have the right metadata
            assert self.mutates_metadata


@dataclasses.dataclass
class SubclassCreationMeta:
    """
    Used for AOTDispatch.
    This dataclass gives us the information we need to reconstruct a tensor subclass
    from our flat inputs.
    Why is this important? The graph that we'd like to trace out contains flat tensor inputs,
    But the user's original model may have subclass inputs and outputs.
    So we need to wrap/unwrap subclasses as necessary to translate between the user's
    view (subclass inps/outs), and the backend compiler's view (graph with no subclass args).

    Complications arise mostly from the fact that a subclass can hold more than one inner tensor;
    So for a given subclass input/output, we need to carefully track which indices map
    to the subclass tensor in the corresponding "dense-tensor-only" graph.
    """

    # In the inner graph that only takes in dense tensor inputs,
    # this maps to the first index of "tensors that should go in this subclass wrapper"
    flat_tensor_start_idx: int
    # The number of tensors that live in this subclass wrapper
    arg_count: int
    # Stores the original subclass itself.
    # This is needed because we need the autograd metadata on the original subclass
    # (this is guaranteed to be a wrapper subclass that holds a fake tensor,
    #  so holding onto this at runtime shouldn't leak memory)
    original_subclass: torch.Tensor
    # meta and inner_keys are produced by the subclass's __tensor_flatten__.
    # We need to keep them around to plumb them into __tensor_unflatten__.
    meta: Any
    inner_keys: List[any]

    def creation_fn(self, all_args, *, is_runtime: bool):
        curr_args = all_args[self.flat_tensor_start_idx:self.flat_tensor_start_idx + self.arg_count]
        assert len(curr_args) == len(self.inner_keys), f'inner_keys: {str(self.inner_keys)}. len(curr_args): {len(curr_args)}'
        out = type(self.original_subclass).__tensor_unflatten__(dict(zip(self.inner_keys, curr_args)), self.meta)
        if not is_runtime:
            # After wrapping up the inner dense tensors into a subclass, we need to make sure that our new wrapper
            # has correct autograd metadata, since we'll be tracing through the autograd engine with the subclass.
            # We don't trace through the autograd engine at runtime though, so no need
            # to compute this extra metadata then!
            torch._mirror_autograd_meta_to(self.original_subclass, out)

        return out

    def __post_init__(self):
        # sanity assert to make sure we don't leak memory
        assert is_fake(self.original_subclass)

# This class encapsulates all aliasing + mutation info we need about the forward graph
# See a more detailed overview of the edge case handling at
# https://docs.google.com/document/d/19UoIh_SVrMy_b2Sx5ZaeOJttm6P0Qmyss2rdBuyfoic/edit
@dataclass(eq=False)
class ViewAndMutationMeta:
    # length = # user inputs
    # This gives us info about every input, and what sort of mutation happened to it (if any)
    input_info: List[InputAliasInfo]

    # length = # user outputs
    # This gives us info about every output (mostly around whether it aliases other tensors)
    output_info: List[OutputAliasInfo]

    # length = the number of intermediate bases appended as outputs to the end of the forward graph.
    # Note: this is not necessarily the same thing as:
    #   len([x for x in output_info if x.output_type == OutputType.alias_of_intermediate])
    # Because outputs might share a ._base, or an output's ._base might itself be
    # another user output (in both cases, we won't redundantly append bases to the end of the graph)
    num_intermediate_bases: int

    # For inference only: instructs us to keep data-only input mutations directly in the graph
    keep_input_mutations: int

    # length = (# inputs w data mutations) + (# user outputs that are non_aliasing tensors)
    #        + (# intermediate bases)
    # These are the FakeTensor (or potential SymInt) outputs that we traced from our
    # metadata pass of the user's forward function.
    # Their only use today is to pass them as a best-guess for tangents when tracing the joint.
    # Stashing them as part of our "metadata" makes it simpler if we want to run our analysis
    # pass once, and re-use the output throughout AOTAutograd
    traced_tangents: List[Any]

    # Each of these is a list telling us about subclasses for the inputs/outputs/grad_outs
    # They are used throughout AOTDispatch to tell us how to generate a list of subclass tensors,
    # Given a (potentially larger) list of plain torch tensors.

    # Taking subclass_inp_meta as an example:
    #   subclass_inp_meta[i] = j (an int) tells us:
    #     "The i'th user input is not a subclass, and corresponds to inputs[j] of the plain-tensor graph."
    #   subclass_inp_meta[i] = SubclassCreationMeta(flat_tensor_start_idx=3, arg_count=2)
    #     "The i'th user input is subclass holding two inner tensors, which are
    #      inputs[3] and inputs[4] of the plain-tensor graph".

    # length = # user inputs
    subclass_inp_meta: List[Union[int, SubclassCreationMeta]]
    # So, the full set of outputs to the forward graph looks something like:
    # (*mutated_inps, *user_outs, *intermediate_bases, *saved_for_bw_tensors)
    # where the first 3 of those 4 can be subclasses
    # (but not saved_for_bw tensors, since these are internal to the compiler
    # and not user visible, so there's no point in wrapping/unwrapping them at runtime).
    # This list contains subclass information on all of the fw graph outputs
    # except for saved_for_bw_tensors.
    subclass_fw_graph_out_meta: List[Union[int, SubclassCreationMeta]]
    # length = # backward graph inputs
    subclass_tangent_meta: List[Union[int, SubclassCreationMeta]]
    # TODO: we should kill this
    # (need to default it to not break internal)
    is_train: bool = False

    num_symints_saved_for_bw: Optional[int] = None

    # The grad_enabled mutation that will be emitted in the runtime_wrapper epilogue
    # NOTE: AOTAutograd will assume that the ambient `is_grad_enabled` is the grad mode
    # that is intended to be in effect prior to running the graph, in keeping with
    # equivalence to eager mode. It is the responsibility of upstream graph acquisition
    # to reset the grad mode to its pre-graph value prior to calling aot_autograd.
    grad_enabled_mutation: Optional[bool] = None

    def __post_init__(self):
        mutated_inp_indices = [
            i for i, m in enumerate(self.input_info) if m.mutates_metadata or m.mutates_data
        ]
        # pre-compute the indices of the inputs that are mutated.
        # When keep_input_mutations is set, we don't need to worry about our epilogue
        # handling data-only mutations, because we keep them directly in the graph.
        mutated_inp_runtime_indices = [
            i for i, m in enumerate(self.input_info) if m.mutates_metadata or (not self.keep_input_mutations and m.mutates_data)
        ]
        aliased_out_indices = [
            i
            for i, m in enumerate(self.output_info)
            if m.output_type not in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]
        ]
        unsafe_view_out_indices = [
            i for i, m in enumerate(self.output_info) if m.output_type is OutputType.unsafe_view_alias
        ]

        self.mutated_inp_indices = mutated_inp_indices
        # This is pre-computed in post_init for perf.
        # It contains the index of every element
        # of input_info that corresponds to a mutation (data or metadata or both)
        self.mutated_inp_runtime_indices = mutated_inp_runtime_indices
        # This is pre-computed for perf.
        # It contains the index of every element
        # of output_info that corresponds to an alias (either of an input or intermediate)
        self.aliased_out_indices = aliased_out_indices
        self.unsafe_view_out_indices = unsafe_view_out_indices
        self.num_outputs = len(self.output_info)
        self.num_outputs_non_aliased = len(
            [x for x in self.output_info
             if x.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]]
        )
        self.num_outputs_aliased_to_inputs = len(
            [
                x
                for x in self.output_info
                if x.output_type in [
                    OutputType.alias_of_input,
                    OutputType.is_input,
                ]
            ]
        )
        self.num_unsafe_view_outputs = len(self.unsafe_view_out_indices)
        self.num_outputs_aliased_to_intermediates = len(
            [
                x
                for x in self.output_info
                if x.output_type in [
                    OutputType.alias_of_intermediate,
                    OutputType.alias_of_intermediate_save_as_output,
                    OutputType.alias_of_intermediate_base_is_user_output,
                ]
            ]
        )
        self.num_outputs_aliased = (
            self.num_outputs_aliased_to_inputs + self.num_outputs_aliased_to_intermediates
        )
        self.num_mutated_data_inputs = len(
            [x for x in self.input_info if x.mutates_data]
        )
        self.num_mutated_metadata_inputs = len(
            [
                x
                for x in self.input_info
                if x.mutates_metadata
            ]
        )
        self.num_mutated_metadata_only_inputs = len(
            [
                x
                for x in self.input_info
                if not x.mutates_data and x.mutates_metadata
            ]
        )
        self.num_mutated_inputs = self.num_mutated_data_inputs + self.num_mutated_metadata_only_inputs
        self.dynamic_outputs = any(
            o.dynamic_dims for o in self.output_info
        )
        # See Note: [AOTAutograd Backward Guards]
        # This is pre-computed for fast asserts on the types of our grad_outputs in the backward.
        # Eventually, we should kill this and replace with real backward guards.
        # (we want to precompute the "runtime" types, so replace FakeTensor with torch.Tensor)
        self.output_types = [torch.Tensor if isinstance(x, FakeTensor) else type(x) for x in self.traced_tangents]

        self.is_rng_op_functionalized = config.functionalize_rng_ops
        # All of the above metadata is collected by tracing the fw function.
        # However, extra outputs for rng offsets behave differently. Both fwd
        # and bwd graphs have their own outputs for the total consumed offsets.
        # Unlike mutated inputs, we don't have to worry about sending the right
        # set of tensors between fwd and bwd. Fwd and bwd offsets are
        # independent and simpler to handle. Therefore, we track them
        # separately.
        self.num_outputs_rng_offset = 1 if self.is_rng_op_functionalized else 0

        # Our forward() returns both (mutated_inputs, outputs, output_intermediate_bases, saved_tensors, saved_symints)
        self.num_forward_returns = self.num_mutated_inputs + self.num_outputs + self.num_intermediate_bases
        # In case of functionalization of rng ops, the fw_module returns one
        # additional output for rng offset. This rng offset is used right
        # away to advance the rng state, and is not passed on to the raw
        # outputs. However, we need to know the exact boundary to identify
        # which tensors to be saved for the bwd graph.  num_forward captures
        # this information.
        self.num_forward = self.num_forward_returns + self.num_outputs_rng_offset

    @property
    def tensors_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None
        if self.num_symints_saved_for_bw > 0:
            return slice(self.num_forward, -self.num_symints_saved_for_bw)
        else:
            return slice(self.num_forward, None)

    @property
    def symints_saved_for_backwards_slice(self):
        assert self.num_symints_saved_for_bw is not None
        if self.num_symints_saved_for_bw > 0:
            return slice(-self.num_symints_saved_for_bw, None)
        else:
            return slice(0, 0)  # empty slice

    def __eq__(self, other):
        if not isinstance(other, ViewAndMutationMeta):
            return NotImplemented
        return (self.input_info == other.input_info and
                self.output_info == other.output_info and
                self.num_intermediate_bases == other.num_intermediate_bases and
                self.keep_input_mutations == other.keep_input_mutations and
                self.is_rng_op_functionalized == other.is_rng_op_functionalized and
                self.num_outputs_rng_offset == other.num_outputs_rng_offset and
                len(self.traced_tangents) == len(other.traced_tangents) and
                all(x.shape == y.shape and x.dtype == y.dtype for x, y, in zip(self.traced_tangents, other.traced_tangents)))

@dataclass(eq=False)
class SubclassMeta:
    # A copy of all forward metadata, but computed on the *dense* tensor forward (after desugaring subclasses)
    # So for example, if the user had a model containing two `TwoTensor` inputs,
    # Then `SubclassMeta.fw_metadata.input_infos` would have length 4 here.
    fw_metadata: ViewAndMutationMeta

    # Note: [Computing Subclass Metadata about grad_inputs]
    # Given a list of flattened, plain tensor grad_inputs, this tells us how to reconstruct the grad_input subclasses
    #
    # You might think: why not just assume that all grad_inputs will have the same subclass-ness as the original inputs?
    # (AOTAutograd generally assumes other properties, e.g. that grad_outputs are contiguous)
    #
    # This doesn't really work though. take this example:
    #
    # def f(DoubleTensor, DenseTensor):
    #     return DoubleTensor  * DenseTensor
    #
    # In the above example, the .grad field of *both* DoubleTensor and DenseTensor will be a DoubleTensor.
    # When we trace out a joint fw-bw graph, we'll end up returning two subclasses for the two grad_inputs.
    # This means that our backward graph will return 4 outputs (two dense tensors for each DoubleTensor grad_input)
    # and we need to properly store the metadata that tells us how to turn these 4 outputs back into DoubleTensors.
    #
    # Note that this info **cannot** easily be figured out from ViewAndMutationMeta.
    # We can only compute this info by tracing the entire joint and examining the grad_inputs that we computed.
    #
    # See Note: [AOTAutograd Backward Guards]
    # This will also eventually require us to install backward guards,
    # in case we made incorrect assumptions about the subclass-ness of our grad_outputs
    #
    # Optional field because we don't compute for inference graphs
    grad_input_metas: Optional[List[Union[int, SubclassCreationMeta]]]

    def __init__(self):
        # The fields in this class get set after its construction.
        pass


# This class exists because:
# - the autograd.Function.forward() in aot autograd returns outputs that might alias inputs
# - we only care about the metadata on those aliases, so we can regenerate them.
#   We do not want them to participate in the autograd.Function.
# We do that by wrapping them in an opaque class, so the autograd.Function
# does not know to treat them as tensors.
@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor


def has_same_metadata(t1, t2):
    return (
        definitely_true(sym_eq(t1.size(), t2.size()))
        and definitely_true(sym_eq(t1.stride(), t2.stride()))
        and definitely_true(t1.storage_offset() == t2.storage_offset())
        and definitely_true(t1.storage_offset() == t2.storage_offset())
        and t1.is_conj() == t2.is_conj()
        and t1.is_neg() == t2.is_neg()
    )


def gen_alias_from_base(aliased_base_tensor, target_meta_tensor, target_requires_grad):
    # Try to do view-replay if possible.
    # fall back to .as_strided() if we can't.
    if target_meta_tensor._base is not None:
        # The base that we want to replay our view off of might have a different shape than the view's original base.
        b = target_meta_tensor._base
        abt = aliased_base_tensor
        # Don't unnecessarily call as_strided if nothing changed; as_strided's
        # backward is poorly implemented and slow
        if abt is not b and (
            abt.size() != b.size() or
            abt.stride() != b.stride() or
            abt.storage_offset() != b.storage_offset()
        ):
            reshaped_base_tensor = aliased_base_tensor.as_strided(
                b.size(), b.stride(), b.storage_offset()
            )
        else:
            reshaped_base_tensor = aliased_base_tensor
        out = target_meta_tensor._view_func(reshaped_base_tensor)
        # This shape mismatch can happen due to a bug in inplace/view handling in autograd.
        # Try putting a breakpoint here and running
        # `test/functorch/test_aotdispatch TestAOTAutograd.test_output_all_alias_types`
        # Also, https://github.com/pytorch/pytorch/issues/49825
        #
        # As a stopgap, we'll fall back to as_strided.
        if out is not None and out.shape == target_meta_tensor.shape:
            if aliased_base_tensor.requires_grad and not target_requires_grad:
                out = out.detach()
            elif not aliased_base_tensor.requires_grad and target_requires_grad:
                out.requires_grad_(True)
            return out
    size = target_meta_tensor.size()
    stride = target_meta_tensor.stride()
    storage_offset = target_meta_tensor.storage_offset()
    if aliased_base_tensor.is_complex() and not target_meta_tensor.is_complex():
        aliased_out = torch.view_as_real(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    elif not aliased_base_tensor.is_complex() and target_meta_tensor.is_complex():
        aliased_out = torch.view_as_complex(aliased_base_tensor).as_strided(
            size, stride, storage_offset
        )
    else:
        aliased_out = aliased_base_tensor.as_strided(size, stride, storage_offset)
    # For outputs aliasing inputs, we need to check if the requires-gradness has changed.
    if aliased_base_tensor.requires_grad and not target_requires_grad:
        aliased_out = aliased_out.detach()
    elif not aliased_base_tensor.requires_grad and target_requires_grad:
        aliased_out.requires_grad_(True)
    return aliased_out

def to_fun(t):
    if isinstance(t, Tensor):
        if is_traceable_wrapper_subclass(t):
            # See Note [Functionalization always runs last]
            # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
            # goes at the bottom.
            # recurse here, so we can support nested wrapper subclasses
            out = transform_subclass(t, lambda _, inner_t: to_fun(inner_t))
            torch._mirror_autograd_meta_to(t, out)
            return out
        else:
            return FunctionalTensor.to_functional(t)
    else:
        return t

def sync_functional_tensor(t):
    if is_traceable_wrapper_subclass(t):
        attrs, ctx = t.__tensor_flatten__()
        for attr in attrs:
            sync_functional_tensor(getattr(t, attr))
    else:
        torch._sync(t)

# When subclasses are involved, t here will usually look something like:
# SubclassA(SubclassB(FunctionalTensor(_to_functional_tensor(FakeTensor))))
def from_fun(t):
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # See Note [Functionalization always runs last]
        # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
        # goes at the bottom.
        # recurse here, so we can support nested wrapper subclasses
        out = transform_subclass(t, lambda _, inner_t: from_fun(inner_t))
        torch._mirror_autograd_meta_to(t, out)
        return out

    if not isinstance(t, FunctionalTensor):
        # quick sanity assert
        if isinstance(t, torch.Tensor):
            assert not torch._is_functional_tensor(t)
        return t
    sync_functional_tensor(t)
    return torch._from_functional_tensor(t.elem)

def is_fun(t):
    if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
        # See Note [Functionalization always runs last]
        # This means that if we want to "functionalize" a subclass, we need to ensure that the functional wrapper
        # goes at the bottom.
        # recurse here, so we can support nested wrapper subclasses
        t_attrs, _ = t.__tensor_flatten__()
        t_inners = [getattr(t, attr) for attr in t_attrs]
        any_fun = any(is_fun(x) for x in t_inners)
        all_fun = all(is_fun(x) for x in t_inners)
        assert any_fun == all_fun
        return any_fun

    return isinstance(t, FunctionalTensor)

# t here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
def has_data_mutation(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        # A tensor subclass was updated if any of its inner elements were updated
        return any(has_data_mutation(getattr(t, attr)) for attr in attrs)
    else:
        if isinstance(t, torch.Tensor):
            assert isinstance(t, FunctionalTensor)
            return torch._functionalize_has_data_mutation(t.elem)
        return False

def are_all_mutations_hidden_from_autograd(t):
    if is_traceable_wrapper_subclass(t):
        attrs, _ = t.__tensor_flatten__()
        # If all inner elements are mutations hidden from autograd, then it is a mutation hidden from autograd.
        return all(are_all_mutations_hidden_from_autograd(getattr(t, attr)) for attr in attrs)
    elif isinstance(t, torch.Tensor):
        assert isinstance(t, FunctionalTensor)
        return torch._functionalize_are_all_mutations_hidden_from_autograd(t.elem)
    else:
        return False

# f_arg here is either
# (1) A FunctionalTensor(_to_functional_tensor(FakeTensor))
# (2) A traceable tensor subclass that holds a FunctionalTensor
# Assumption: arg promises to be the "original" tensor wrapped by f_arg
def has_metadata_mutation(f_arg, arg, *, check_storage_mutation: bool):
    if is_traceable_wrapper_subclass(f_arg):
        attrs, _ = f_arg.__tensor_flatten__()
        # A tensor subclass was updated if any of its inner elements were updated
        f_inner_ts = [getattr(f_arg, attr) for attr in attrs]
        inner_ts = [getattr(arg, attr) for attr in attrs]
        return any(has_metadata_mutation(f_inner_t, inner_t, check_storage_mutation=check_storage_mutation)
                   for f_inner_t, inner_t in zip(f_inner_ts, inner_ts))
    else:
        if not isinstance(f_arg, torch.Tensor):
            assert not isinstance(arg, torch.Tensor)
            return False
        assert isinstance(f_arg, FunctionalTensor)
        assert isinstance(arg, FakeTensor)

        arg_after = torch._from_functional_tensor(f_arg.elem)
        # This is true if the current tensor experienced at least one set_() call
        maybe_storage_changed = torch._functionalize_was_storage_changed(f_arg.elem)
        # However, multiple set_() calls can cancel out. So we also check whether the
        # storage of the tensor has changed.
        # Note: if an input experienced two set_() calls that cancel out, **and**
        # it experiences an data mutation, we pessimistically think that the set_()
        # call is necessary here. We could in theory fix this, but this will
        # hopefully never happen in user code, and is not needed for fsdp.
        same_storages = StorageWeakRef(arg.untyped_storage()) == StorageWeakRef(arg_after.untyped_storage())
        has_storage_metadata_mutation = maybe_storage_changed and not same_storages
        if check_storage_mutation:
            return has_storage_metadata_mutation

        # This is true if the current tensor experienced at least one metadata mutation.
        maybe_metadata_mutated = torch._functionalize_has_metadata_mutation(f_arg.elem)
        # However, multi metadata mutations can cancel out.
        # So we also check if the concrete sizes/strides on the tensor have changed.
        same_sizes = arg.shape == arg_after.shape
        same_strides = arg.stride() == arg_after.stride()
        same_offsets = arg.storage_offset() == arg_after.storage_offset()
        has_metadata_mutation_ = maybe_metadata_mutated and not (same_sizes and same_strides and same_offsets)
        # We consider a tensor to have been metadata mutated if its storage was mutated through a set_() call.
        return has_storage_metadata_mutation or has_metadata_mutation_


def _get_hints(exprs):
    """
    Get the hints of a list/tuple of int/SymInt.
    """
    if isinstance(exprs, (list, tuple)):
        return type(exprs)(_get_hints(e) for e in exprs)
    elif isinstance(exprs, torch.SymInt):
        return exprs.node.shape_env.size_hint(exprs.node.expr)
    else:
        return exprs

def requires_subclass_dispatch(args, fw_metadata: ViewAndMutationMeta) -> bool:
    args_flattened = pytree.arg_tree_leaves(*args)
    any_subclass_args = any(is_traceable_wrapper_subclass(x) for x in args_flattened if isinstance(x, Tensor))
    any_subclass_outputs = any(is_traceable_wrapper_subclass(x) for x in fw_metadata.traced_tangents if isinstance(x, Tensor))
    # This tells us whether or not we need to perform any unwrapping/wrapping of tensor subclasses at runtime.
    return any_subclass_args or any_subclass_outputs

# Given a flat list of arguments, some of which may be tensor subclasses,
# computes metadata about "how to reconstruct the current list of subclasses,
# if we were given their flattened dense tensors instead"
def create_subclass_meta(curr_args: List[Any]) -> List[Union[int, SubclassCreationMeta]]:
    idx = 0
    infos = []
    for a in curr_args:
        if isinstance(a, torch.Tensor) and is_traceable_wrapper_subclass(a):
            attrs, meta = a.__tensor_flatten__()
            start_idx = idx
            cnt = len(attrs)
            curr_cnt = cnt
            infos.append(SubclassCreationMeta(
                flat_tensor_start_idx=start_idx,
                arg_count=curr_cnt,
                original_subclass=a,
                meta=meta,
                inner_keys=attrs,
            ))
        else:
            infos.append(idx)
            cnt = 1
        idx += cnt
    return infos

def _get_autocast_states():
    return [
        torch.is_autocast_enabled(),
        torch.is_autocast_cpu_enabled(),
        torch.get_autocast_gpu_dtype(),
        torch.get_autocast_cpu_dtype(),
        torch.is_autocast_cache_enabled(),
    ]


# This is a version of functionalization that is specifically designed
# for the AOTAutograd use case.
#
# Unlike functorch's variant, this doesn't use the functorch level system,
# instead it directly uses PyTorch's conventional dispatcher to hit the
# functionalization key.  In particular, this means that FunctionalTensorWrapper
# can have autograd data stored directly on it.
#
# In typical AOTAutograd usage, the dispatch key order will look like:
#
#   Autograd - Functionalization ~~~~> Proxy Mode - Fake Tensor
#       outer tensor                        inner tensor
#
# Returns:
# - ViewAndMutationMeta, telling us metadata about the inputs and outputs, and
#   The list of outputs from the forward, but **only** the outputs that we need
#   to pass in as tangents into the backward.
#   Specifically, aliased outputs from the forward get regenerated, and don't participate
#   in the compiled backward function.
def run_functionalized_fw_and_collect_metadata(
    f,
    *,
    keep_input_mutations: bool,
    # TODO: refactor to kill this flag
    is_train: bool = False,
) -> ViewAndMutationMeta:
    memo = {}

    def _to_fun(t):
        if isinstance(t, Tensor):
            if t in memo:
                return memo[t]
            r = to_fun(t)
            memo[t] = r
            return r
        else:
            return t

    @wraps(f)
    def inner(*flat_args):
        # This function is meant to be run with the forward, which expects a flat list of tensor/symint/other args.
        assert all(isinstance(a, KNOWN_TYPES) for a in flat_args)

        input_info: List[InputAliasInfo] = []
        output_info: List[OutputAliasInfo] = []

        flat_f_args = pytree.tree_map(_to_fun, flat_args)

        prior_grad_enabled = torch.is_grad_enabled()
        prior_autocast_states = _get_autocast_states()

        # See Note [Disabling Functionalize TLS Above Python Functionalization]
        disable_above = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
        with disable_above, FunctionalTensorMode():
            # precondition: The passed in function already handles unflattening inputs + flattening outputs
            flat_f_outs = f(*flat_f_args)

        if prior_autocast_states != _get_autocast_states():
            raise RuntimeError(
                "AOTAutograd does not support tracing graphs that mutate the autocast state. "
                "Dynamo will only insert autocast context managers (e.g. with torch.autocast(..)) into the graph, "
                "which will unwind all of their mutations to autocast state before the graph exits. "
                "If you encounter this error while using torch.compile, please file a bug."
            )

        # Inspect the state of the input tensor functional wrapper to detect input mutation info
        # If inp[i] has a metadata-only mutation, then maybe_inputs_with_mutated_metadata[i] contains the updated version
        for (i, (arg, f_arg)) in enumerate(zip(flat_args, flat_f_args)):
            if not isinstance(arg, Tensor):
                new_arg = arg
            else:
                new_arg = from_fun(f_arg)
            mutates_metadata = has_metadata_mutation(f_arg, arg, check_storage_mutation=False)
            mutates_storage_metadata = has_metadata_mutation(f_arg, arg, check_storage_mutation=True)
            mutates_data = has_data_mutation(f_arg)
            mutations_hidden_from_autograd = are_all_mutations_hidden_from_autograd(f_arg)

            # Here, we're saying that if an input experienced a set call, inp.set_(other),
            # then we can effectively not have to worry about whether its data was mutated.
            # There are 3 cases:
            # (1) We mutate inp *after* the set_() call. other is a graph intermediate.
            #     In this case, we're not really mutating the input storage of "inp";
            #     we're mutating the storage of an intermdiate value (other),
            #     and slamming that storage into the input tensor. So no data mutation is necessary.
            # (2) We mutate inp *after* the set_() call. other is a graph *input*.
            #     In this case, the data mutation will be properly handled in the runtime
            #     epilogue during the processing of "other"
            # (3) We mutate inp *before* the set_() call.
            #     This case is *not* currently handled.
            #     TODO: discuss this in the PR. Both supporting this, and detecting + erroring out,
            #     seem painful to get working.
            if mutates_storage_metadata:
                mutates_data = False

            input_info.append(InputAliasInfo(
                is_leaf=isinstance(arg, torch.Tensor) and safe_is_leaf(arg),
                mutates_data=mutates_data,
                mutates_metadata=mutates_metadata,
                mutations_hidden_from_autograd=mutations_hidden_from_autograd,
                mutates_storage_metadata=mutates_storage_metadata,
                requires_grad=isinstance(f_arg, torch.Tensor) and f_arg.requires_grad,
            ))

        # If a function involves creating a tensor, and returning a view of it, such that its _base is the intermediate,
        # We need to make sure our graph returns the _base as a graph output, and we manually recreate the view
        # to return to the user. Why? The backend compiler is free to (incorrectly) not set requires_grad
        # on the base tensor, but we are obligated to properly set requires-gradness on the real output.

        inp_storage_refs = {
            StorageWeakRef(inpt.untyped_storage()): idx
            for idx, inpt in enumerate(flat_f_args)
            if isinstance(inpt, torch.Tensor)
        }

        # We need inp tensor id's to be able to tell if an outputs **are** inputs.
        inp_tensor_ids = {
            id(inpt) for inpt in flat_f_args if isinstance(inpt, torch.Tensor)
        }
        # We need output tensor id's to tell if any output._base` attributes **are** other outputs.
        # (This is also a dict because we need to know that output's index, so we can regenerate
        # the alias from it).
        out_tensor_ids = {id(o): i for i, o in enumerate(flat_f_outs)}

        # Keep track of which outputs alias other outputs
        out_tensor_alias_counts = collections.defaultdict(int)
        # This tells us, for a given group of outputs that alias each other,
        # whether they e.g. all came from an unbind call
        num_aliased_tensors_that_are_multi_output_views = collections.defaultdict(int)
        out_storage_to_tensors = collections.defaultdict(set)
        for o in flat_f_outs:
            if isinstance(o, torch.Tensor):
                curr_storage = StorageWeakRef(o.untyped_storage())
                out_tensor_alias_counts[curr_storage] += 1
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                # This is an optimization on top of the "alias of intermediates" logic,
                # which you can read more about under Note [AOT Autograd: outputs aliasing inputs or intermediates!]
                #
                # Before describing the optimization: this is important for AOTAutograd to have good
                # perf around, multi-output views. HOWEVER:
                # - There is a more generic change to AOTAutograd that we'd like to make, that subsumes this case,
                #   around using pre-dispatch tracing to partition out a graph so we can faithfully replay all
                #   views without having to regenerate them at runtime.
                # - It's loosely described in this doc (more details will be added soon):
                #   https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit
                # - Once that change lands, we should just rip out this "optimization", since:
                #   (1) It will be fully unnecessary
                #   (2) Although it is only a few lines of code, it is a bit difficult to reason about
                #       its correctness with the autograd engine in all cases.
                #
                #
                # What is this optimization? Consider the below case:
                # def f(x):
                #     intermediate = x.mul(2)
                #     # x and intermediate here require grad
                #     o1, o2, ... o10 = intermediate.unbind(-1)
                #     return intermediate, o1, o2, ... o10
                # Now, the "intermediate base" handling in AOTAutograd implies that we must do the following:
                #   (1) return "intermediate as an extra output of the compiled graph
                #   (2) regenerate each aliased output off of "intermediate", **outside** of the autograd.Function.
                # The reason AOTAutograd ordinarily does this is for safety: the autograd engine needs to know
                # that o1 through o10 are all aliased, and if we blindly return o1 through o10 from the autograd.Function,
                # this information will be hidden.
                # In particular, mutating one alias might require autograd to update autograd metadata on the other aliases
                # (like their grad_fn, for example, when the autograd engine needs to do view-replay).
                #
                # However, intermediate_base logic can be bad for backward performance (we sometimes generate
                # as_strided calls during the intermediate base logic, which can have a slow backward formula).
                # Is it possible to find a set of conditions where it is **safe** to hide the output aliasing from autograd?
                #
                # For a set of outputs of the graph that alias each other, o_1...o_k, consider:
                # (1) They came from the same multi-output view op, e.g. o_1, ..., o_k = intermediate.unbind(0)
                # (2) If there are any other aliases of o_1 through o_k (in the example above, intermediate),
                #     **at most** 1 can escape from the graph (e.g. there is not some other graph input/output
                #     o_other, that aliases these outputs)
                # (3) o_1...o_k all require_grad, they all share the same ._base, and their ._base requires grad.
                #     This condition is important because it's what causes slowness in the intermediate_base
                #     codepath of aot_autograd. Ordinarily, o_1...o_k would all get a grad_fn, and
                #     aot_autograd's view-replay might give each output an AsStridedBackward as its grad_fn.
                #     "K" AsStridedBackward calls will be *much* slower than a single UnbindBackward.
                # In this setup, is it possible to mutate one of the outputs o_i in a way that would affect the autograd meta
                # of the other aliases?
                #
                # Claim: No! Consider a few example (which I'm pretty sure cover all cases of mutation w.r.t. autograd):
                # (a) What happens if we mutate any of o_1 through o_k directly?
                #     Autograd raises an error:
                #     "RuntimeError: Output 0 of UnbindBackward0 is a view and is being modified inplace. This view is
                #      the output of a function that returns multiple views. Such functions do not allow the output
                #      views to be modified inplace. You should replace the inplace operation by an out-of-place one."
                # (b) What if we take a view of o_k and mutate it, o_k.view(o_k.shape).mul_(2)?
                #     Autograd raises the same error- the "multi-output-view"ness of an alias propagates to future views.
                # (c) What if we mutate o_k under no_grad?
                #     Autograd raises the same error
                # (d) What if we detach and mutate, e.g. o_k.detach().mul_(2)?
                #     Autograd allows this, *but* autograd updates all alias's grad_fn's to be error functions when accessed.
                #     Autograd raises the same error
                # (e) What if we try to mutate another alias of o_1...o_k, that was **not** created from a multi-output view?
                #     We promised that there is at most **one** such alias, e.g. intermediate in the example above.
                #     You can mutate intermediate, but in eager mode this will change the grad_fn of o_1...o_k
                #     to be error fn's.
                #     Since intermediate was the *only* non-multi-output-alias, there are no other aliases
                #     of `intermediate` around that were produced by the compiled fn and have a valid grad_fn.
                #
                # Coming back to this optimization:
                # Given that it is not possible for mutating one of these aliases to affect the autograd metadata of another alias
                # without causing an error in eager mode, we will simple hide the aliasing from autograd during torch.compile
                # if all of the above conditions are met.
                # This has the slight downside that it's possible to write some "bad" code that autograd will raise an error on
                # in eager but fail to during torch.compile, but it has the benefit that this code has much better performance.
                # NOTE: if and when we eventually update AOTAutograd to do the "view graph slicing" defined here:
                # https://docs.google.com/document/d/1DlfFq8TKbuAn2zyJxLfoW-X1qkkm5PLdHFtySo03QAk/edit,
                # then this optimization will probably matter less and might be ok to remove.
                is_cur_tensor_multi_out_view = isinstance(o, FunctionalTensor) \
                    and torch._functionalize_is_multi_output_view(o.elem)
                if is_cur_tensor_multi_out_view:
                    num_aliased_tensors_that_are_multi_output_views[curr_storage] += 1
                out_storage_to_tensors[curr_storage].add(o)

        # maps the id of an intermediate base to its index in the output of the compiled forward
        intermediate_base_tensor_id_to_output_idx: Dict[int, int] = {}
        intermediate_bases: List[torch.Tensor] = []
        for o in flat_f_outs:
            curr_storage = None if not isinstance(o, torch.Tensor) else StorageWeakRef(o.untyped_storage())
            outs_with_identical_metadata_that_require_grad = [] if not isinstance(o, torch.Tensor) else [
                curr for curr in out_storage_to_tensors[curr_storage]
                if has_same_metadata(o, curr) and curr.requires_grad and o is not curr
            ]
            is_result_of_custom_autograd_fn = False
            if isinstance(o, torch.Tensor):
                # Need to check for both custom cpp (CppFunction) and python (BackwardCFunction) autograd fns
                if type(o.grad_fn).__name__ == "CppFunction":
                    is_result_of_custom_autograd_fn = True
                if isinstance(o.grad_fn, torch.autograd.function.BackwardCFunction):
                    is_result_of_custom_autograd_fn = True

            if not isinstance(o, torch.Tensor):
                output_type = OutputType.non_alias
                base_idx = None
            elif curr_storage in inp_storage_refs and o.grad_fn is not None \
                    and is_result_of_custom_autograd_fn:
                output_type = OutputType.custom_function_view
                base_idx = None
            elif curr_storage in inp_storage_refs:
                base_idx = inp_storage_refs[curr_storage]
                is_input_tensor = id(o) in inp_tensor_ids
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = num_aliased_tensors_that_are_multi_output_views[curr_storage]
                num_aliased_outs_that_are_not_multi_output_views = num_aliased_outs - num_multi_output_view_outs
                if o.grad_fn is not None and num_aliased_outs_that_are_not_multi_output_views == 0:
                    # See Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                    # In particular, given:
                    # def f(x):
                    #     return list(x.unbind(0))
                    # The main reason we ordinarily try to regenerate these output aliases outside of the
                    # compiled autograd.Function is because if any of the outputs are later mutated,
                    # autograd needs to perform view-replay to regenerate them.
                    # However, autograd does not allow users to mutate multi-output views
                    # in any way that can change the autograd metadata of other aliases.
                    # So we hide this aliasing from autograd here.
                    aot_graphs_log.info("Encountered AOTAutograd case: differentiable outputs that \
alias each other from a multi-output view call")
                    output_type = OutputType.non_alias
                elif is_input_tensor:
                    output_type = OutputType.is_input
                else:
                    output_type = OutputType.alias_of_input

            # We only need to handle the intermediate base case when both
            # the intermediate base and the output require gradients.
            # See Note [AOT Autograd: outputs aliasing inputs or intermediates!]
            elif (
                o._base is not None
                and o.requires_grad
                and o._base.requires_grad
            ):
                num_aliased_outs = out_tensor_alias_counts[curr_storage]
                num_multi_output_view_outs = num_aliased_tensors_that_are_multi_output_views[curr_storage]
                num_aliased_outs_that_are_not_multi_output_views = num_aliased_outs - num_multi_output_view_outs
                # Note: [AOTAutograd: differentiable outputs that alias each other from a multi-output view call]
                if out_tensor_alias_counts[curr_storage] == 1 or num_aliased_outs_that_are_not_multi_output_views <= 1:
                    # Note [Intermediate Bases Optimization]
                    # Normally if we have an output that aliases an intermediate,
                    # we need to add the extra "intermediate base" logic further down
                    # to prevent autograd from yelling at us if the user later tries to
                    # mutate that output.
                    # However, the common case here is if we have an output that aliases an intermediate,
                    # but doesn't alias any other outputs.
                    # In that case, autograd shouldn't have to worry about the aliasing at all
                    # (if that output is mutated, there are no other live aliases for autograd to worry about).
                    # The "intermediate bases" can hurt inductor perf by forcing more variables to become outputs.
                    # So as an optimization, we won't do intermediate base handling in this case.
                    # Instead, we'll hide the aliasing from autograd using aten._unsafe_view().
                    if out_tensor_alias_counts[curr_storage] != 1 and num_aliased_outs_that_are_not_multi_output_views <= 1:
                        aot_graphs_log.info("Encountered AOTAutograd case: differentiable outputs that alias each other \
from a multi-output view call")
                    output_type = OutputType.unsafe_view_alias
                    base_idx = None
                else:
                    # First, check if o's ._base is an existing output
                    maybe_existing_out_idx = out_tensor_ids.get(id(o._base), None)
                    if maybe_existing_out_idx is not None:
                        # Special case where the output is an alias of a graph intermediate, but that intermediate
                        # is itself also a user output.
                        output_type = OutputType.alias_of_intermediate_base_is_user_output
                        base_idx = maybe_existing_out_idx
                    else:
                        # Next, check if o's ._base is an intermediate base that we already returned
                        maybe_existing_base_output_idx = intermediate_base_tensor_id_to_output_idx.get(
                            id(o._base), None
                        )
                        if maybe_existing_base_output_idx is not None:
                            output_type = OutputType.alias_of_intermediate
                            base_idx = maybe_existing_base_output_idx
                        else:
                            # Otherwise, take o._base and explicitly return it as an output in the compiled graph
                            new_out_idx = len(intermediate_bases)
                            base_idx = new_out_idx
                            # Indicate to the logic later on (when we trace the joint)
                            # that this particular output should get it's ._base appended to the forward graph outputs
                            output_type = OutputType.alias_of_intermediate_save_as_output
                            intermediate_base_tensor_id_to_output_idx[id(o._base)] = new_out_idx
                            intermediate_bases.append(o._base)
            elif (
                # See https://github.com/pytorch/pytorch/issues/100348 for this case.
                # This protects against the specific case where a user fn returns (output, output.detach())
                out_tensor_alias_counts[curr_storage] > 1
                and len(outs_with_identical_metadata_that_require_grad) > 0
                and not o.requires_grad
            ):
                assert len(outs_with_identical_metadata_that_require_grad) > 0
                # In theory we could use any of these tensors to regenerate the aliased outputs from,
                # since they all alias each other and have identical metatadata
                out_alias = outs_with_identical_metadata_that_require_grad[0]
                existing_out_idx = out_tensor_ids[id(out_alias)]
                output_type = OutputType.alias_of_intermediate_base_is_user_output
                base_idx = existing_out_idx
            else:
                output_type = OutputType.non_alias
                base_idx = None

            if isinstance(o, torch.Tensor):
                dynamic_dims = {i for i, s in enumerate(o.shape) if not is_concrete_int(s)}
            else:
                dynamic_dims = None
            out_info = OutputAliasInfo(
                output_type=output_type,
                raw_type=type(o),
                base_idx=base_idx,
                dynamic_dims=dynamic_dims,
                requires_grad=isinstance(o, torch.Tensor) and o.requires_grad
            )
            output_info.append(out_info)

        # See Note [AOT Autograd: Views to avoid tangents aliasing inputs]
        def view_avoid_dupes_with_primals(t):
            if isinstance(t, Tensor) and is_traceable_wrapper_subclass(t):
                return transform_subclass(t, lambda _, inner_t: view_avoid_dupes_with_primals(inner_t))
            if isinstance(t, Tensor):
                return t.view(t.shape)
            return t

        # This analysis function returns *only* the outputs that are meant to be tangents to the backwards.
        # Anything that aliases (inputs returned in the fw due to metadata mutations, or outputs that alias inputs/intermediates)
        # are *regenerated* later, and not used directly in the autograd graph
        f_input_tangents = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutates_data and info.requires_grad
        ]
        f_output_tangents = [
            o
            for o, info in zip(flat_f_outs, output_info)
            if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]
            and issubclass(info.raw_type, torch.Tensor)
            and info.requires_grad
        ]
        # intermediate bases are also included in the backward graph
        f_tangents = f_input_tangents + f_output_tangents + intermediate_bases
        traced_tangents = pytree.tree_map(from_fun, f_tangents)
        traced_tangents = pytree.tree_map(view_avoid_dupes_with_primals, traced_tangents)
        user_outs = pytree.tree_map(from_fun, f_output_tangents)

        f_mutated_inputs = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutates_data or info.mutates_metadata
        ]
        f_metadata_mutated_inputs = [
            inp
            for inp, info in zip(flat_f_args, input_info)
            if info.mutates_metadata
        ]
        # This logic (annoyingly) re-figures out exactly what the outputs to the compiled fw graph will be.
        # When handling subclasses, we need info about **all** outputs of compiled forward graph,
        # so we know precisely which graph outputs to wrap back into tensor subclasses
        # Ideally we would refactor this so not have an is_train flag, and have the separate
        # inference and training paths decide which inputs/output to ask for subclass info on.
        # However, we currently stash indexing information on each SubclassMeta about its order
        # in the graph outputs list.
        f_fw_graph_outs = list(flat_f_outs)
        if is_train or not keep_input_mutations:
            f_fw_graph_outs = f_mutated_inputs + f_fw_graph_outs
        else:
            # even when "keep_input_mutations" is True,
            # we never keep metadata-only mutations in the fw graph
            f_fw_graph_outs = f_metadata_mutated_inputs + f_fw_graph_outs
        if is_train:
            f_fw_graph_outs = f_fw_graph_outs + intermediate_bases
        fw_graph_outs = pytree.tree_map(from_fun, f_fw_graph_outs)

        grad_enabled_mutation = None
        if torch.is_grad_enabled() != prior_grad_enabled:
            grad_enabled_mutation = torch.is_grad_enabled()
            torch.set_grad_enabled(prior_grad_enabled)  # Restore the prior state after tracing it
            aot_graphs_log.info(
                ("grad_mode mutation encountered in graph. "
                 "Will emit mutation epilogue, to set grad_mode=%s"),
                grad_enabled_mutation
            )

        metadata = ViewAndMutationMeta(
            input_info=input_info,
            output_info=output_info,
            num_intermediate_bases=len(intermediate_bases),
            keep_input_mutations=keep_input_mutations,
            traced_tangents=traced_tangents,
            subclass_inp_meta=create_subclass_meta(flat_args),
            subclass_fw_graph_out_meta=create_subclass_meta(fw_graph_outs),
            subclass_tangent_meta=create_subclass_meta(traced_tangents),
            is_train=is_train,
            grad_enabled_mutation=grad_enabled_mutation,
        )
        return metadata

    return inner


@dataclass
class BackwardSignature:
    """
    Provides information about the backward section of an exported
    joint forward-backward graph.
    For a particular fx GraphModule, this class contains information on:
    (1) A mapping from each gradient (backwards output) to the parameter
        it corresponds to (forward input)
    (2) A mapping from each gradient (backwards output) to the user input
        it corresponds to (forward input)
    (3) Which of the forward outputs corresponds to the loss, that we backprop on.

    Each string name is the `node.name` of the corresponding node in the fx graph.
    """
    gradients_to_parameters: Dict[str, str]
    gradients_to_user_inputs: Dict[str, str]
    loss_output: str

GraphOutputName = NewType('GraphOutputName', str)
GraphInputName = NewType('GraphInputName', str)
FQN = NewType('FQN', str)

@dataclass
class GraphSignature:
    """
    Provides information about an exported module.
    For a particular fx GraphModule, this class contains information on:
    (1) Which graph inputs are parameters, buffers, or user inputs
    (2) (for params/buffers) a mapping from the name of each graph argument
        to its parameter/buffer FQN in the original nn.Module.
    (3) If there are input mutations, these are represented as extra outputs
        in the fx GraphModule. We provide a mapping from these
        extra output names to the names of the actual inputs.
    (4) The pytree metadata on how to flatten/unflatten inputs and outputs.
        The corresponding FX GraphModule only accepts and returns
        pytree-flattened inputs/outputs.
    (5) (Optionally) if the FX is a joint forward-backward graph, we provide
        a signature on the backward section of the joint graph.
    """

    parameters: List[FQN]
    buffers: List[FQN]

    user_inputs: List[GraphInputName]
    user_outputs: List[GraphOutputName]
    inputs_to_parameters: Dict[GraphInputName, FQN]
    inputs_to_buffers: Dict[GraphInputName, FQN]

    # If the user's module mutates a buffer,
    # it's represented in the graph as an extra graph output.
    # This dict is a mapping from
    # "graph outputs that correspond to updated buffers"
    # to the FQN names of those mutated buffers.
    buffers_to_mutate: Dict[GraphOutputName, FQN]

    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec

    backward_signature: Optional[BackwardSignature]

    @classmethod
    def from_tracing_metadata(
        cls,
        *,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
        graph_input_names: List[str],
        graph_output_names: List[str],
        view_mutation_metadata: ViewAndMutationMeta,
        named_parameters: List[str],
        named_buffers: List[str],
        num_user_inputs: int,
        num_user_outputs: int,
        loss_index: Optional[int],
        backward_signature: Optional[BackwardSignature],
    ) -> "GraphSignature":
        graph_inputs = graph_input_names
        graph_outputs = graph_output_names
        parameters = list(named_parameters)
        buffers = list(named_buffers)

        # Calling convention assumptions:
        # (1) graph inputs = (params, buffers, user_inputs)
        # (2) graph outputs = (mutated_inputs, user_outs, param_gradients)
        # (If we are capturing an inference graph, this convention is identical
        #  except that param_gradients is empty)
        user_inputs = graph_inputs[len(parameters) + len(buffers) :]
        assert num_user_inputs == len(user_inputs)
        assert len(graph_inputs) == (len(parameters) + len(buffers) + len(user_inputs))

        inputs_to_parameters = dict(zip(graph_inputs[: len(parameters)], parameters))
        inputs_to_buffers = dict(zip(
            graph_inputs[len(parameters) : len(parameters) + len(buffers)],
            buffers,
        ))

        state_names = [*parameters, *buffers]
        mutated_buffers = []
        for idx, input_info in enumerate(view_mutation_metadata.input_info):
            if input_info.mutates_data:
                # Only buffers can be mutated, not parameters
                assert idx >= len(parameters)
                buffer_name = state_names[idx]
                mutated_buffers.append(buffer_name)

        assert len(mutated_buffers) == view_mutation_metadata.num_mutated_inputs

        start, stop = 0, view_mutation_metadata.num_mutated_inputs
        buffers_to_mutate = dict(zip(graph_outputs[start:stop], mutated_buffers))

        start, stop = stop, stop + num_user_outputs
        user_outputs = graph_outputs[start:stop]

        unused_outputs = len(graph_outputs) - stop
        if backward_signature is not None:
            unused_outputs -= len(backward_signature.gradients_to_parameters) + len(
                backward_signature.gradients_to_user_inputs
            )
        assert unused_outputs == 0

        return GraphSignature(
            parameters=parameters,
            buffers=buffers,
            user_inputs=user_inputs,
            user_outputs=user_outputs,
            inputs_to_buffers=inputs_to_buffers,
            inputs_to_parameters=inputs_to_parameters,
            buffers_to_mutate=buffers_to_mutate,
            in_spec=in_spec,
            out_spec=out_spec,
            backward_signature=backward_signature,
        )

@dataclasses.dataclass
class AOTConfig:
    """
    Configuration for AOTDispatcher
    """

    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: Dict[Callable, Callable]
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = False
    no_tangents: bool = False
    dynamic_shapes: bool = False
    aot_autograd_arg_pos_to_source : Optional[List[Source]] = None
    inference_compiler: Optional[Callable] = None
    enable_log: bool = True

# This function takes in a tensor t, and returns one of t, t.view(), or t.clone().
# When tracing the joint forward + backward, for any inputs in the graph that are mutated,
# we need to clone them first (and similarly for metadata-only mutations, we need to view them first).
# The idea is that when we trace the backward, we need to pass in the *original* primals
# to autograd.grad(), before they were mutated.
# Note: when we have synthetic base inputs, we need to clone them *before* creating views off of them.
# This means that "idx" here represents the index of the (potentially) synthetic base.
# What we need to do is:
# (1) map the current (post-synthetic-base calling convention) input argument index
#     to int index pre-synthetic-base-calling-convention.
# (2) There could be multiple, if this index corresponds to a synthetic base
#     that has multiple input aliases.
# (3) If any of those corresponding inputs get metadata mutations, then we clone the base.
def maybe_to_fresh_input(idx, t, meta):
    if not isinstance(t, Tensor):
        return t
    if idx in meta.mutated_inp_indices:
        # We only need to bother cloning mutated inputs that participate in autograd.
        mutated_inp_idx = meta.mutated_inp_indices.index(idx)
        if meta.input_info[idx].requires_grad and meta.input_info[idx].mutates_data:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the mutation
            return t.clone()
        # No need to do anything for  meta.input_info[idx].mutates_storage_metadata,
        # Because autograd doesn't support set_()
        if meta.input_info[idx] and meta.input_info[idx].mutates_metadata:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the metadata mutation
            return t.view(t.shape)
    return t

# This function returns a new function that returns mutated inputs as outputs.
# if keep_data_input_mutations is set, then we assume that data-only mutations
# will be left in the graph, and we only return metadata-mutated inputs as outputs.
def fn_input_mutations_to_outputs(
    fn: Callable,
    meta: ViewAndMutationMeta,
    keep_data_input_mutations: bool,
) -> Any:
    def inner_fn(*args):
        outs = fn(*args)
        assert len(meta.output_info) == len(outs)
        # The compiled fw will return mutated input tensors, *including* metadata-only mutation.
        # However, if keep_data_input_mutations is set, the compiled fw only needs to return metadata-mutated inputs.
        # (because data-only input mutations are handled directly in the compiled graph)
        mutated_inputs_to_return = [
            x
            for (i, x) in enumerate(args)
            if meta.input_info[i].mutates_metadata or (meta.input_info[i].mutates_data and not keep_data_input_mutations)
        ]
        return *mutated_inputs_to_return, *outs
    return inner_fn

# This function takes in a fn with external aliasing and mutation,
# and returns a new fn with no external aliasing and mutation,
# as needed for autograd.
# The main transformations are:
# - Return mutated inputs as extra outputs
# - Clone mutated inputs that require gradients,
#   because autograd will require us to pass the pre-mutated inputs into autograd.grad
# - Return intermediate bases of outputs as additional outputs,
#   needed to appease autograd.Function
# The new function returns:
# (1) The updated outputs
# (2) A boolean mask of len(new_fn_outputs),
#     that can be used to tell autograd.grad which outputs should get tangents
#     if we trace the backward.
def fn_prepped_for_autograd(
    fn: Callable,
    meta: ViewAndMutationMeta,
) -> Any:
    def inner_fn(*args):
        args_maybe_cloned = [
            maybe_to_fresh_input(i, t, meta) for i, t in enumerate(args)
        ]

        outs = fn(*args_maybe_cloned)
        assert isinstance(outs, (tuple, list))
        outs = list(outs)
        assert len(meta.output_info) == len(outs)

        mutated_inputs_to_return = [
            x
            for (i, x) in enumerate(args_maybe_cloned)
            if meta.input_info[i].mutates_metadata or meta.input_info[i].mutates_data
        ]

        intermediate_bases = []
        for i, (o, info) in enumerate(zip(outs, meta.output_info)):
            if info.output_type == OutputType.alias_of_intermediate_save_as_output:
                intermediate_bases.append(o._base)

        assert meta.num_intermediate_bases == len(intermediate_bases)

        # the compiled forward should return (mutated_inputs, user_outs, intermediate_bases)
        fw_outs_to_return = *mutated_inputs_to_return, *outs, *intermediate_bases

        # Also return a boolean mask specifying which outputs to this function will be used as tangents
        mutated_inputs_grad_mask = [
            meta.input_info[meta.mutated_inp_indices[i]].mutates_data and meta.input_info[meta.mutated_inp_indices[i]].requires_grad
            for (i, x) in enumerate(mutated_inputs_to_return)
        ]

        # Pass any (non-aliased) outputs in as tangents, since they'll be returned as outputs in the fw
        # For outputs that are aliases of intermediates, we will have returned the output's _base as an output in the graph instead,
        # which we *should* send to grad()
        output_grad_mask = [
            meta.output_info[i].output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]
            # Also, only tensor outputs should participate in the backward
            # (in particular, Symint outputs in the forward graph shouldn't get tangents)
            and issubclass(meta.output_info[i].raw_type, torch.Tensor)
            and meta.output_info[i].requires_grad
            for (i, x) in enumerate(outs)
        ]

        intermediate_base_grad_mask = [True for _ in range(len(intermediate_bases))]

        out_grad_mask = mutated_inputs_grad_mask + output_grad_mask + intermediate_base_grad_mask
        assert len(out_grad_mask) == len(fw_outs_to_return)

        # Take care to grab and sync the updated inputs from primals_after_cloning (the inputs we actually mutate!)
        # and not primals (the preserved inputs, pre-mutation, that we pass to grad())
        # This is annoying: our joint function needs to be aware of functionalization
        # (syncing mutated inputs before calling autograd.grad())
        # In theory, we could make the autograd engine do this automatically, although that probably isn't any cleaner.
        for arg in args_maybe_cloned:
            if not isinstance(arg, Tensor):
                continue
            sync_functional_tensor(arg)

        return fw_outs_to_return, out_grad_mask
    return inner_fn

# Given a fn, computes the joint.
# NOTE: fn is expects the following behavior:
# (1) fn() needs to return a tuple of (outs, mask),
#     where `mask` tells us which outputs are meant to have tangents.
#     we don't know this info automatically, because we don't actually want to blindly
#     compute tangents for every output that requires grad.
#     Specifically, outputs that alias inputs won't participate in the backward and get tangents.
# (2) fn() cannot mutate any inputs that require gradient.
#     otherwise, when we compute autograd.grad(), we will not take those input mutations into account
#     (the way this is handled is that we ensure any inputs that normally get mutated are cloned first)
def create_joint(
    fn: Callable, *, aot_config: AOTConfig
) -> Any:
    def inner_fn(primals: List[Any], tangents: List[Any]):
        outs, tangent_mask = fn(*primals)
        assert len(tangent_mask) == len(outs)
        outs_to_grad = [o for needs_tangent, o in zip(tangent_mask, outs) if needs_tangent]
        assert len(outs_to_grad) == len(tangents)

        # Get the inputs that need gradients
        grad_primals = []
        inputs_needs_grads = []
        # Note that we're not using primals here,
        # being carefully not to pass any mutated inputs into autograd.grad()
        for p in primals:
            is_grad_tensor = isinstance(p, Tensor) and p.requires_grad
            inputs_needs_grads.append(is_grad_tensor)
            if is_grad_tensor:
                grad_primals.append(p)

        # Get the outputs that need gradients
        needed_outs = []
        needed_tangents = []
        for out, tangent in zip(outs_to_grad, tangents):
            if isinstance(out, Tensor) and out.requires_grad:
                # A bit sketchy, but fixes e.g. test_aot_autograd_exhaustive_matmul_cpu_float32
                # The issue is that we are sensitive to decomps that don't accurately maintain
                # their output's _base.shape compared to eager mode, and this helps mitigate a bit.
                # The not definitely_false is also sketchy; if unbacked
                # symints are involved, we're just going to assume that the
                # decomps setup the base shape correctly
                needed_outs.append(
                    out if not definitely_false(sym_eq(out.shape, tangent.shape)) else out.view(tangent.shape)
                )
                needed_tangents.append(tangent)

        setup_stacktrace_preservation_hooks([out.grad_fn for out in needed_outs])

        if config.functionalize_rng_ops:
            PhiloxStateTracker.mark_beginning_of_backward()
        backward_out = []
        # Call the backwards pass
        if grad_primals:
            with fx_traceback.preserve_node_meta():
                # for full graph export, we always export a joint graph where we assume no tangents are needed.
                if aot_config.no_tangents:
                    assert len(needed_tangents) == 1 and needed_tangents[0].numel() == 1
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        allow_unused=True,
                    )
                else:
                    backward_out = torch.autograd.grad(
                        needed_outs,
                        grad_primals,
                        grad_outputs=needed_tangents,
                        allow_unused=True,
                    )
        backward_out_iter = iter(backward_out)
        return outs, [
            next(backward_out_iter) if i else None for i in inputs_needs_grads
        ]

    def inner_fn_with_anomaly(*args):
        with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Anomaly Detection has been enabled."
            )
            with torch.autograd.detect_anomaly(check_nan=False):
                return inner_fn(*args)

    return inner_fn_with_anomaly

# This creates the final function that we want to trace using make_fx(),
# in both aot_dispatch_autograd and aot_dispatch_base.
# Preconditions:
# - fn corresponds to the user's fw function
# - fn arguments have been flattened, duplicate arguments have been handled
# - In the returned function, the "primals" arguments *includes* synthetic bases.
# This function does the work of functionalizing the input function,
# and performing copy_() calls at the end of the function if `keep_input_mutations` is set.
# The function returned has signature that is either:
# (1) "traced_fn(primals: List[Any])" if trace_joint is False
# (2) "traced_fn(primals: List[Any], tangents: List[Any])" if trace_joint is True
# Returns a new (functionalized) function, and updated arguments to call it with.
def create_functionalized_fn(
    fn,
    args,
    *,
    meta: ViewAndMutationMeta,
    aot_config: AOTConfig,
    trace_joint: bool,
) -> Tuple[Callable, List[Any]]:
    def functionalized_f_helper(*args):
        # Wrap inputs into functional wrappers
        f_args = pytree.tree_map(to_fun, args)

        # See Note [Disabling Functionalize TLS Above Python Functionalization]
        disable_above = torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))
        with disable_above, FunctionalTensorMode():
            # Run the joint
            f_outs = fn(*f_args)

        if aot_config.keep_inference_input_mutations and not trace_joint:
            # Note: This is a bit annoying. There's a layering issue here, where:
            # (1) functionalization needs to operate on **synthetic base** inputs, before unpacking them into the "real" inputs.
            # (2) For keep_input_mutations, we support tracing a call to copy_() directly on mutated inputs.
            #     However, we **only** want to support this for inputs that have data-only (and no metadata) mutations,
            #     because inductor (and backends in generally) would prefer not to see these (e.g. as_strided_(), resize_()).
            #     This makes it pretty difficult for this logic to operate on synthetic bases.
            # (3) In addition, there are cases where it's significantly cheaper to perform the copy on the individual
            #     (unpacked) input aliases, instead of the synthetic base.
            # Example case where (3) could be important:
            #
            #     def f(x, y):
            #         x.mul_(2)
            #         y.mul_(3)
            #         return x, y
            #    a = torch.ones(1'000'000)
            #    x, y = out(a[0:9], a[1:10])
            #
            # It would be much better to add copy_() calls into the graph for the two tiny slices, instead of materializing
            # a giant "updated synthetic base" and copying into a's entire storage.
            #
            # For now, we are pessimistically not performing the optimization from (3);
            # we will materialize an "updated" synthetic base, and copy it back to the synthetic input base.
            # This allows us to factor aot autograd much more nicely, since only one area of the code needs to worry
            # about synthetic bases.
            for i, (inpt_old, inpt_f) in enumerate(zip(args, f_args)):
                if not isinstance(inpt_f, torch.Tensor):
                    continue
                assert is_fun(inpt_f)
                inpt_new = from_fun(inpt_f)
                if meta.input_info[i].mutates_data and not meta.input_info[i].mutates_metadata:
                    # We found an input that had a (data-only) mutation.
                    # Since keep_input_mutations is set, we need to faithfully apply a copy_()
                    # so the compiler will see the input mutation in the graph.
                    assert inpt_new is not inpt_old
                    if meta.input_info[i].mutations_hidden_from_autograd:
                        with torch.no_grad(), torch.autograd._unsafe_preserve_version_counter(inpt_old):
                            inpt_old.copy_(inpt_new)
                    else:
                        inpt_old.copy_(inpt_new)

        return pytree.tree_map(from_fun, f_outs)

    # Kinda annoying, but needed to make sure that the fx graph we trace out has "primals"
    # and "tangents" as its input names (which are special-cased by the partitioner)
    def joint_helper(primals, tangents):
        return functionalized_f_helper(primals, tangents)

    def fwd_helper(*args):
        return functionalized_f_helper(*args)

    helper = joint_helper if trace_joint else fwd_helper
    if config.functionalize_rng_ops:
        # Setup the wrapper for functionalization of rng ops
        helper, args = create_functionalized_rng_ops_wrapper(helper, args, trace_joint)

    return helper, args

def create_graph(f, args, *, aot_config: AOTConfig) -> torch.fx.GraphModule:
    with enable_python_dispatcher():
        fx_g = make_fx(f, decomposition_table=aot_config.decompositions)(*args)

    return fx_g


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


aot_autograd_decompositions = {}


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: List[str] = []
# TODO: It would be nice to reset the numbering every time aot_id goes
# up, but this is annoying to do right now (because we don't know if
# an aot_id will come back from the dead), so right now this also happens
# to be a globally unique number too (at the cost of wobbling if you change
# how the graphs compile)
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name):
    global model_name
    model_name = name


def get_aot_compilation_context() -> Tuple[List[str], str, int]:
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    """
    Returns the name of the graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}__{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(aot_config, graph_name):
    global graph_being_compiled
    # TODO: Don't shove the aot_id in here; set it in the context
    graph_being_compiled = [f"{aot_config.aot_id}_{graph_name}"]
    try:
        yield
    finally:
        global nth_graph
        nth_graph += 1
        graph_being_compiled = []


def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True
    return g


def make_boxed_compiler(compiler):
    @wraps(compiler)
    def f(fx_g, inps):
        out_f = compiler(fx_g, inps)
        fx_g = make_boxed_func(out_f)
        return fx_g

    return f


def call_func_with_args(f, args, steal_args=False, disable_amp=False):
    if not steal_args:
        args = list(args)
    assert isinstance(args, list)

    context = torch._C._DisableAutocast if disable_amp else nullcontext
    with context():
        if hasattr(f, "_boxed_call"):
            out = normalize_as_list(f(args))
        else:
            # TODO: Please remove soon
            # https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
            warnings.warn(
                "Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. "
                "Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. "
                "See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale."
            )
            out = normalize_as_list(f(*args))
    return out

def aot_dispatch_base_graph(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta
) -> Tuple[Callable, List[Any], Optional[SubclassMeta]]:
    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
    # The cases that aot_dispatch_base doesn't need to handle include:
    # - outputs that are aliases of graph intermediates
    # - outputs that are aliases of graph inputs
    # While cases that it does need to handle include:
    # - input mutations (including when inputs are aliases of each other)
    # - input metadata mutations
    fn_to_trace = fn_input_mutations_to_outputs(
        flat_fn,
        fw_metadata,
        keep_data_input_mutations=aot_config.keep_inference_input_mutations,
    )

    fn_to_trace, updated_flat_args = create_functionalized_fn(
        fn_to_trace, flat_args, meta=fw_metadata, aot_config=aot_config, trace_joint=False)

    fn_to_trace, updated_flat_args_subclasses_desugared, maybe_subclass_meta = aot_dispatch_subclass(
        fn_to_trace, updated_flat_args, is_joint_structure=False, meta=fw_metadata, fw_only=flat_fn)

    fw_module = create_graph(
        fn_to_trace,
        updated_flat_args_subclasses_desugared,
        aot_config=aot_config,
    )

    # As long as we opted to remove input mutations, then
    # there should be *NO* mutating ops in the graph at this point.
    copy_count = assert_functional_graph(fw_module.graph, allow_input_mutations=aot_config.keep_inference_input_mutations)

    fw_module.graph.eliminate_dead_code()
    fw_module.recompile()

    copy_count2 = assert_functional_graph(fw_module.graph, allow_input_mutations=aot_config.keep_inference_input_mutations)

    assert copy_count == copy_count2

    if aot_config.enable_log:
        aot_graphs_log.info("%s", lazy_format_graph_code("Forward graph", fw_module, aot_config.aot_id))

    # TODO: should factor this into a separate function for export that always only returns just the graph.
    if aot_config.is_export:
        assert maybe_subclass_meta is None, "aot_export_module does not support tensor subclass inputs for now."
        return fw_module
    return fw_module, list(updated_flat_args_subclasses_desugared), maybe_subclass_meta

def aot_dispatch_base(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    fw_module, updated_flat_args, maybe_subclass_meta = aot_dispatch_base_graph(
        flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)

    disable_amp = torch._C._is_any_autocast_enabled()
    context = torch._C._DisableAutocast if disable_amp else nullcontext

    with context(), track_graph_compiling(aot_config, "inference"):
        compiler = aot_config.inference_compiler if aot_config.inference_compiler is not None else aot_config.fw_compiler
        if config.functionalize_rng_ops:
            # Add the seed and offset as example inputs to pass to the compiler
            fake_mode = detect_fake_mode()
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
            updated_flat_args.extend([seed, offset])

        if torch._guards.TracingContext.get():
            torch._guards.TracingContext.get().fw_metadata = fw_metadata \
                if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
        compiled_fw = compiler(fw_module, updated_flat_args)

    # This boxed_call handling happens inside create_runtime_wrapper as well.
    # However, create_runtime_wrapper does not expect the rng offsets in the
    # output. So, we have to create another wrapper and take out the offset. As
    # a result, we have to account for not boxed_call compilers as well.
    if not hasattr(compiled_fw, "_boxed_call"):
        compiled_fw = make_boxed_func(compiled_fw)

    # Create a wrapper to set up the rng functionalize bits
    @wraps(compiled_fw)
    def rng_functionalization_wrapper(args):
        # args is a list because compiled_fw is boxed_call
        if fw_metadata.is_rng_op_functionalized:
            # Add the seed and offset to args
            seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
            args.extend([seed, offset])
            out = compiled_fw(args)
            out = functionalized_rng_runtime_epilogue(fw_metadata, out)
            return out
        else:
            return compiled_fw(args)

    if maybe_subclass_meta is not None:
        compiled_fw_func = aot_dispatch_subclass_wrapper(
            rng_functionalization_wrapper, subclass_metas=fw_metadata.subclass_fw_graph_out_meta, num_fw_outs_saved_for_bw=None)
    else:
        compiled_fw_func = rng_functionalization_wrapper

    if not hasattr(compiled_fw_func, "_boxed_call"):
        compiled_fw_func = make_boxed_func(compiled_fw_func)

    compiled_fn = create_runtime_wrapper(
        compiled_fw_func,
        runtime_metadata=fw_metadata,
        indices_of_inps_to_detach=[],
        trace_joint=False,
        keep_input_mutations=aot_config.keep_inference_input_mutations,
        disable_amp=disable_amp
    )

    return compiled_fn


# Returns the number of detected copy_
def assert_functional_graph(fx_g: torch.fx.Graph, *, allow_input_mutations: bool = False) -> int:
    placeholders = set()
    copy_count = 0
    # NB: It would also be nice to verify that the mutations all happen at the
    # end, but we also do some administrative views after mutations so this
    # isn't actually true.  (TODO: Could this cause problems for Inductor?)
    for n in fx_g.nodes:
        if n.op == "placeholder":
            placeholders.add(n)
        if isinstance(n.target, torch._ops.OpOverload):
            if n.target is aten.copy_.default and allow_input_mutations:
                suffix = True
                # Can only copy_ into an input, and can only do so once
                assert n.args[0] in placeholders
                placeholders.remove(n.args[0])
                copy_count += 1
            else:
                assert not n.target._schema.is_mutable, \
                    f'aot_autograd expected to have an entirely functional graph, but found {n.format_node()}'
    return copy_count


def are_differentiable_views(view1, view2):
    if view1 is view2:
        return True
    if view1._base is None and view2._base is None:
        return False
    if view1._base is view2._base or view1._base is view2 or view1 is view2._base:
        return True
    return False


def same_dtype_views(view1, view2):
    if view1.dtype != view2.dtype:
        return False
    if view1._base is not None and view1.dtype != view1._base.dtype:
        return False
    if view2._base is not None and view2.dtype != view2._base.dtype:
        return False
    return True



# Assumption: x and y are known to share a storage, and we are trying to determine
# if their memory is actually completely disjoint, based on sizes/strides/storage_offset
def tensors_definitely_do_not_overlap(x, y):
    if x is y:
        return False
    if x.numel() == 0 or y.numel() == 0:
        return True

    # Make x always on the left
    if x.storage_offset() > y.storage_offset():
        x, y = y, x
    # Short-circuit in the "obvious" overlapping case: both tensors are contiguous
    if x.is_contiguous() and y.is_contiguous():
        if x.storage_offset() + x.numel() > y.storage_offset():
            # definitely overlap
            return False
        else:
            # definitely no overlap
            return True

    if x.dim() == 2 and y.dim() == 2 and x.stride(1) == 1 and y.stride(1) == 1:
        # This cases is needed for the shampoo optimizer.
        # All tensors are 2d (non-contiguous), have the same outer stride, and have an inner stride of 1
        # (so rows are contiguous)
        if x.stride(0) == y.stride(0):
            offset_delta = y.storage_offset() - x.storage_offset()
            if offset_delta < x.size(1):
                # definitely overlaps (row 0 of y overlaps with row 0 of x)
                # Example:
                #   base = torch.arange(32).reshape(4, 8)
                #   x = base.narrow(1, 0, 4)
                #     x: size=(4, 4), stride=(8, 1), offset=0
                #   y = base.narrow(1, 3, 4)
                #     y: size=(4, 4), stride=(8, 1), offset=3
                return False
            x_total_elems_covered = x.stride(0) * (x.size(0) - 1) + x.size(1)
            if x_total_elems_covered <= offset_delta:
                # definitely does not overlap (last byte of x is before start of y)
                # Example:
                #   x: size=(4, 4), stride=(8, 1), offset=0 (last byte is 27)
                #   y: size=(4, 4), stride=(8, 1), offset=28 (start byte is 28)
                return True
            # At this point, we want to check if the 0th row of y
            # overlaps with **some** row of x.
            # We can check this by shifting y backward by the shared stride, repeatedly,
            # until the first row of y is before the first row of x.
            # Then we can check if these rows overlap.
            # We can accomplish this by modding our offset by the stride.
            offset_delta_mod = offset_delta % x.stride(0)
            # Example:
            # 0 1 2 3
            # 9 10 11 12
            # 18 19 20 21
            # 27 28 29 30
            #   x: size=(4, 4), stride=(9, 1), offset=0
            #   y: size=(4, 4), stride=(9, 1), offset=22 (this would not overlap)
            #   y: size=(4, 4), stride=(9, 1), offset=23 (this would not overlap)
            #   y: size=(4, 4), stride=(9, 1), offset=24 (this would overlap)
            #   y: size=(4, 4), stride=(9, 1), offset=25 (this would overlap)
            # If the interval [modded_offset, modded_offset + x_size] falls entirely
            # without
            if offset_delta_mod + y.size(1) <= x.stride(0):
                return True
            else:
                return False
    return False


def compute_overlapping_inputs(fwd_inputs, aliased_input_indices):
    actual_aliased_indices = set()
    for j in range(len(aliased_input_indices)):
        for i in range(j):
            i_ = aliased_input_indices[i]
            j_ = aliased_input_indices[j]
            if not tensors_definitely_do_not_overlap(fwd_inputs[i_], fwd_inputs[j_]):
                actual_aliased_indices.add(i_)
                actual_aliased_indices.add(j_)
    return actual_aliased_indices

# Note [Handling mutations on an input that aliases other inputs]
# The easiest example to show-case this edge case is here:
#
# def f(a, b):
#     a.mul_(2)
#     out = a + b
#     return out
# b = torch.ones(...)
# a = b.view(-1)
# f(a, b)
#
# In this situation, if a and b happened to be aliased, we need to trace something different!
# Suppose we had b = a.view(-1)
# (In this case, that means that `a._base is b`)
#
# We need to ensure that the aliasing relationship between a and b is preserved.
# We do that detecting the specific situation above (mutate an input that aliases another input),
# and when we do that, we create a synthetic base argument. Then inside of the traced forward,
# we regenerate a and b off of that base.
# The complete example of the transformed function looks like this:
#
# // The traced forward takes in a synthetic base, and regenerates the aliased inputs as views
# // We could consider getting view-replay support here to minimize as_strided_scatter ops in the graph
# def traced_forward(base):
#     a = base.as_strided(...)
#     b = base.as_strided(...)
#     a_updated = a.mul(2)
#     base_updated = torch.as_strided_scatter(base, a_updated, ...)
#     b_updated = base_updated.as_strided(...)
#     out = a_updated + b_updated
#     return a_updated, out
#
# def compiled_fn(a, b):
#     // we detect that a is the "differentiable base" here
#     base = a
#     // In other situations, we might do either:
#     // (1) a and b are both views off of some larger differentiable base
#     //     assert a._base is b._base and a._base is not None
#     //     base = a._base
#     // (2) a and b both don't require gradients. Create a base from the storage
#     //     assert a._base is None and b._base is None
#     //     base = torch.Tensor(a.storage())
#     a_updated, out = traced_forward(base)
#     a.copy_(a_updated)
#     return out
#
# This function:
# (1) Merges input views into a synthetic base argument, when any of those input views are mutated
# (2) Returns metadata telling the autograd.Function how to modify their arguments properly,
#     to respect the new calling convention.
#
# The calling convention is as follows.
# Any inputs that were originally views of one another get yanked, and replaced with a synthetic base.
# The argument list ordering goes [base1, ..., baseN], [arg1, ..., argN],
# Where the ordering of the bases is determined from the ordering of the original view args.
# baseA will come before baseB if the earliest original argument coming from baseA
# showed up earlier in the argument list than the earliest original argument coming from baseB.
#
# Example, given some tensors a, b, c, d
# call site:
#   f(a, c.view(-1), b.view(-1), b, c, d)
# Modified argument list:
#   c_base comes first because the first c view came earlier in arg list than the first b view
#   a and d still show up in the modified arg list, but b and c don't- they're regenerated from their bases
#   b_base = torch.Tensor(b.storage())
#   c_base = torch.Tensor(c.storage())
#   f(c_base, b_base, a, d)
def merge_view_inputs(
    fwd_inputs: List[Any], mutated_input_info: List[InputAliasInfo],
    *,
    # The autograd case currently has more restrictions than the inference case.
    is_inference: bool,
) -> Tuple[List[Any], Optional[List[Union[int, Tuple[int, torch.Tensor]]]]]:
    assert len(fwd_inputs) == len(mutated_input_info)
    storage_ref_to_idx: Dict[StorageWeakRef, List[int]] = collections.defaultdict(list)
    base_args = []
    other_args = []
    for i, inpt in enumerate(fwd_inputs):
        if isinstance(inpt, Tensor):
            storage_ref = StorageWeakRef(inpt.untyped_storage())
            storage_ref_to_idx[storage_ref].append(i)
        else:
            other_args.append(inpt)
    # Note [Synthetic Base Info Metadata]
    # This list contains metadata that tells you what the i'th argument in the inner calling convention should be.
    # It's either:
    # - another int (corresponding to the index in the argument list of the element from the outer calling convention)
    # - idx, view_tensor, where we can generate the new output with view_tensor._view_func(old_args[idx])
    #   idx corresponds to which synthetic base from the outer calling context to view
    inner_calling_convention_meta: Dict[int, Union[int, Tuple[int, torch.Tensor]]] = {}
    for aliased_input_indices in storage_ref_to_idx.values():
        if len(aliased_input_indices) <= 1 or not any(
            # We only care about mutations that affect all aliases,
            # so metadata mutations on an input doesn't require us to do synthetic base handling.
            mutated_input_info[inpt_idx].mutates_data
            for inpt_idx in aliased_input_indices
        ):
            for curr_idx in aliased_input_indices:
                other_args.append(fwd_inputs[curr_idx])
            continue

        # Here, we attempt to do a more complicated check to detect false aliasing
        # (e.g. if all the tensors have the same storage, but don't actually overlap)
        # In theory, we could have a large group of tensors that all share storages, where only *some* of them
        # have overlapping memory.
        # I don't bother with that case for now: here, we only bail out earlier if we detect that **every** pair
        # of tensors in the current group that shares a storage is non-overlapping.
        aliased_input_indices_no_false_sharing = compute_overlapping_inputs(fwd_inputs, aliased_input_indices)
        if len(aliased_input_indices_no_false_sharing) <= 1:
            for curr_idx in aliased_input_indices:
                other_args.append(fwd_inputs[curr_idx])
            continue

        # We detected an input that was mutated, AND aliases with another input.
        # we need to replace this set of aliased inputs with a single synthetic base.
        # For now, I'm banning a bunch of cases. We expect dynamo to properly detect these cases
        # and error out. We can fix them later.
        # These checks are transitive, so we don't need to check every pair.
        for idx1, idx2 in zip(aliased_input_indices, aliased_input_indices[1:], strict=False):
            view1 = fwd_inputs[idx1]
            view2 = fwd_inputs[idx2]
            # The "inputs that are aliased but have different differentiable bases" case
            # is more complicated and hopefully pretty rare. Not currently handled.
            if not is_inference:
                assert are_differentiable_views(
                    view1, view2
                ), "aot_autograd() does not yet handle non-differentiable view input mutations."
            # Regenerating views when reinterpreting complex / real tensors seems non-trivial,
            # not handling for now
            assert same_dtype_views(
                view1, view2
            ), "aot_autograd() does not yet handle input mutations on views with different dtypes."
        non_none_bases = [
            fwd_inputs[i]._base
            for i in aliased_input_indices
            if fwd_inputs[i]._base is not None
        ]
        aliases_with_none_bases = [
            fwd_inputs[i] for i in aliased_input_indices if fwd_inputs[i]._base is None
        ]
        if len(non_none_bases) == 0:
            # Case where none of the aliases have a ._base
            # we generate a synthetic base without gradients, and generate views off of it
            # We hit this case when we have input tensors to the graph that share a storage,
            # but do not have a ._base field.
            # Wondering when we hit this case?
            # The _base field simply says that autograd knows about the aliasing relationship,
            # but sometimes we create tensors which are aliased out of the same storage but guaranteed
            # to be disjoint. In these cases, we will skip setting up the _base relationship
            # for performance reasons (because the fact that the tensors share the same storage
            # is unobservable unless you (1) do naughty things with resize_/as_strided
            # or (2) look at the storage--as we are doing here.)
            # One particular example of this is optimizer steps on the LSTM module:
            # LSTM parameters are packed into a contiguous storage for efficiency reasons when
            # calling cuDNN kernels, so when these parameters get passed to the optimizer we will
            # find they share the same storage, but do not have _base set since they are all disjoint.
            #
            # NOTE: There is one case where this is unsafe:
            # torch.Tensor(storage) will ALWAYS create a 1D tensor, which is not necessarily
            # the same shape as the "actual" base that the tensor came from.
            # For the most part this is fine, because we always use as_strided()
            # to generate the original aliased inputs again.
            # If we were to use view-replay though, this could cause the aliased views
            # to have incorrect sizes.
            example_idx = aliased_input_indices[0]
            example_alias = fwd_inputs[example_idx]
            # Note that this function is re-used at both trace time and runtime.
            # At trace time, we're under a FakeMode so synthetic_base becomes a FakeTensor.
            synthetic_base = torch.empty((0,), dtype=example_alias.dtype, device=example_alias.device)
            # We don't actually have a convenient way of going from storage -> tensor,
            # So using set_() here (we suffer some minor overhead, but this case is rare).
            synthetic_base.set_(example_alias.untyped_storage())
        else:
            # Case where all of the aliases require gradients, and have the same _base.
            synthetic_base = non_none_bases[0]
            for other_base in non_none_bases[1:]:
                assert (
                    other_base is synthetic_base
                ), "aot_autograd() does not yet handle non-differentiable view input mutations."
            for alias in aliases_with_none_bases:
                assert (
                    alias is synthetic_base
                ), "aot_autograd() does not yet handle non-differentiable view input mutations."
        base_args.append(synthetic_base)
        for curr_view_idx in aliased_input_indices:
            curr_view = fwd_inputs[curr_view_idx]
            base_idx = len(base_args) - 1
            # We store just enough info here so that we can regenerate the view later.
            # Regeneration: curr_view._view_func(args[base_idx])
            inner_calling_convention_meta[curr_view_idx] = (base_idx, curr_view)
    if len(base_args) == 0:
        assert len(other_args) == len(fwd_inputs)
        # If no synthetic bases are necessary, just return the original inputs.
        return fwd_inputs, None
    else:
        # Otherwise, return:
        # (1) The new args according to the updated calling convention: (synthetic_bases, other_args)
        # (2) Metadata telling functionalization how to generate the inner argument list given the outer calling convention.
        #     We post-process it into a list, where meta[i] tells you info about the i'th argument in the inner calling convention.
        args_to_functionalization = base_args + other_args
        arg_to_old_idx_map = {arg: i for (i, arg) in enumerate(fwd_inputs)}
        for i, other_arg in enumerate(other_args):
            new_idx = len(base_args) + i
            old_idx = arg_to_old_idx_map[other_arg]
            inner_calling_convention_meta[old_idx] = new_idx
        # post process into a list
        post_processed_calling_convention_meta: List[Union[int, Callable]] = [
            -1 for _ in range(len(inner_calling_convention_meta))
        ]
        for k, v in inner_calling_convention_meta.items():
            post_processed_calling_convention_meta[k] = v
        # Quick assert: every argument in the inner calling convention should be accounted for.
        for x in post_processed_calling_convention_meta:
            assert x != -1
        return args_to_functionalization, post_processed_calling_convention_meta


def format_guard_bug_msg(aot_config, expected):
    return (
        f"At compilation time, graph {aot_config.aot_id} was compiled under the "
        f"assumption that {expected}, but at runtime this was not the case.  "
        "This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."
    )


def remove_dupe_metadata(
    m: ViewAndMutationMeta,
    keep_arg_mask: List[bool],
    add_dupe_map: List[int],
) -> ViewAndMutationMeta:
    assert len(m.input_info) == len(keep_arg_mask)
    # Easy invariant: the first argument should never be a dupe (it will be kept)
    assert len(keep_arg_mask) > 0 and keep_arg_mask[0]

    # Filter dupe'd mutated inputs out of traced_tangents
    num_data_mutations = len([x for x in m.input_info if x.mutates_data])
    other_traced_tangents = m.traced_tangents[num_data_mutations:]
    inp_traced_tangents = m.traced_tangents[:num_data_mutations]
    filtered_inp_traced_tangents = [x for i, x in enumerate(inp_traced_tangents) if keep_arg_mask[m.mutated_inp_indices[i]]]
    traced_tangents = filtered_inp_traced_tangents + other_traced_tangents

    return ViewAndMutationMeta(
        input_info=[x for i, x in enumerate(m.input_info) if keep_arg_mask[i]],
        # For outputs that are views of inputs, we store the index of the input that the output
        # was generated from. Need to update that index to account for removed dupes.
        output_info=[
            OutputAliasInfo(
                output_type=o.output_type,
                raw_type=o.raw_type,
                dynamic_dims=o.dynamic_dims,
                base_idx=None if o.base_idx is None else add_dupe_map[o.base_idx],
                requires_grad=o.requires_grad
            )
            for o in m.output_info
        ],
        num_intermediate_bases=m.num_intermediate_bases,
        keep_input_mutations=m.keep_input_mutations,
        traced_tangents=traced_tangents,
        # We are guaranteed not to get here, since dupes are not supported today with subclass inputs.
        subclass_inp_meta=None,
        subclass_fw_graph_out_meta=None,
        subclass_tangent_meta=None,
        is_train=m.is_train,
    )

# Given our ViewAndMutation metadata, this fn constructs a new set of metadata,
# after adding synthetic base arguments to the function.
# Most of the work in this fn is slogging through all of the metadata corresponding to inputs,
# and updating it with our synthetic base calling convention.
#
# When config.debug_assert is set, we automatically regenerate the metadata
# and compare it to this output for sanity.
#
# In addition to the updated metadata, also return the list of input indices
# that will need to be updated in the synthetic base epilogue
def create_synthetic_base_metadata(
    m: ViewAndMutationMeta,
    # Maps each outer argument idx to its inner idx (or, if this outer arg is generated from a
    # synthetic base, you get a tuple of (i, TensorMeta), telling you the base tensor idx, and view metadata)
    synthetic_base_info: List[Union[int, Tuple[int, torch.Tensor]]],
    outer_args: List[Any],
    inner_args: List[Any],
) -> Tuple[ViewAndMutationMeta, List[int]]:

    S_Outer = NewType('S_Outer', int)
    S_Inner = NewType('S_Inner', int)
    synthetic_base_to_indices: Dict[S_Inner, List[S_Outer]] = {}
    for inner_idx in range(len(inner_args)):
        outer_aliased_indices_of_current_base_arg = [
            outer_idx for outer_idx, inner_idx_or_tuple in enumerate(synthetic_base_info)
            if (isinstance(inner_idx_or_tuple, int) and inner_idx_or_tuple == inner_idx)
            or (isinstance(inner_idx_or_tuple, tuple) and inner_idx_or_tuple[0] == inner_idx)
        ]
        synthetic_base_to_indices[inner_idx] = outer_aliased_indices_of_current_base_arg

    # given the requires_grad info on mutated inputs,
    # generate the requires_grad info on those same mutated inputs, but after constructing synthetic bases.
    input_infos = []
    for outer_indices in synthetic_base_to_indices.values():
        # leaf-ness should be all-or-nothing for aliased tensor.
        # (aka if "a" and "b" are views, then a.is_leaf == b.is_leaf)
        any_leaf = any(m.input_info[x].is_leaf for x in outer_indices)
        all_leaf = all(m.input_info[x].is_leaf for x in outer_indices)
        assert any_leaf == all_leaf
        inpt_info = InputAliasInfo(
            # If len(outer_indices) > 1, then this input is a synthetic base.
            # The invariant is that to the rest of aot autograd, synthetic bases only show up if
            # one of their aliases gets a data mutation. And if any of their aliases get metadata
            # mutations, they will be hidden from the rest of aot autograd.
            mutates_data=True if len(outer_indices) > 1 else m.input_info[outer_indices[0]].mutates_data,
            mutates_metadata=False if len(outer_indices) > 1 else m.input_info[outer_indices[0]].mutates_metadata,
            mutations_hidden_from_autograd=all(m.input_info[x].mutations_hidden_from_autograd for x in outer_indices),
            mutates_storage_metadata=False if len(outer_indices) > 1 else m.input_info[outer_indices[0]].mutates_storage_metadata,
            is_leaf=any_leaf,
            requires_grad=any(m.input_info[x].requires_grad for x in outer_indices)
        )
        input_infos.append(inpt_info)

    # Find any inputs that fulfill the following criteria:
    # (1) They are part of a synthetic base (because they alias another input,
    #      and at least one input experiences a data mutation)
    # (2) They experience a metadata mutation
    outer_aliased_arg_idx_with_metadata_mutations = [
        outer_idx for outer_idx, inpt_info in enumerate(m.input_info)
        if inpt_info.mutates_metadata and not isinstance(synthetic_base_info[outer_idx], int)
    ]

    # grab the original requires grad info on the outputs, except the ones from the mutated inputs
    input_metadata_output_info = [
        OutputAliasInfo(
            output_type=OutputType.alias_of_input,
            raw_type=FunctionalTensor,
            dynamic_dims={i for i, s in enumerate(outer_args[outer_idx].shape) if not is_concrete_int(s)},
            base_idx=synthetic_base_info[outer_idx][0],
            requires_grad=outer_args[outer_idx].requires_grad
        ) for outer_idx in outer_aliased_arg_idx_with_metadata_mutations]
    existing_output_infos = [
        OutputAliasInfo(
            output_type=o.output_type,
            raw_type=o.raw_type,
            dynamic_dims=o.dynamic_dims,
            # Map the input idx pre-synthetic-bases to the new idx post-synthetic-bases
            base_idx=None if o.base_idx is None
            else synthetic_base_info[o.base_idx]
            if isinstance(synthetic_base_info[o.base_idx], int)
            else synthetic_base_info[o.base_idx][0],
            requires_grad=o.requires_grad
        )

        for o in m.output_info]

    inner_mutated_tangents = [
        x
        for inner_idx, x in enumerate(inner_args)
        if input_infos[inner_idx].mutates_data and input_infos[inner_idx].requires_grad
    ]

    output_info = existing_output_infos + input_metadata_output_info
    # Regenerate traced tangents to include mutated inputs including synthetic bases
    traced_tangents = inner_mutated_tangents + m.traced_tangents[len(inner_mutated_tangents):]

    return ViewAndMutationMeta(
        input_info=input_infos,
        output_info=output_info,
        num_intermediate_bases=m.num_intermediate_bases,
        keep_input_mutations=m.keep_input_mutations,
        traced_tangents=traced_tangents,
        # We are guaranteed not to get here, since synthetic_base codepaths are not supported today with subclass inputs.
        subclass_inp_meta=None,
        subclass_fw_graph_out_meta=None,
        subclass_tangent_meta=None,
        is_train=m.is_train,
    ), outer_aliased_arg_idx_with_metadata_mutations

# MOTIVATION:
#
# When tracing functions for future execution, one must be careful not to pass
# in the same input tensor multiple times (e.g., f(x, x), as this can result
# in graphs that are ONLY valid if you later pass a new tensor in exactly the
# same way (e.g., f(y, y)).  (NB: we really mean duplicate; two distinct
# tensors that alias each other is a different situation that is covered by
# aot_dispatch_deduplicated_autograd). Here are two examples:
#
# (1) Suppose you have a function:
#
#   def f(x, y):
#       return x + y
#
# If you make_fx(f)(x, x), you will trace out:
#
#   def f(x, y):
#       return y + y
#
# Oops!
#
# (2) For most tensors x and y, you can compute f's gradient with respect to
# these to inputs by saying torch.autograd.grad(f(x, y), (x, y)).  However,
# if x is y, you will trace out a program that gets incorrect gradients:
#
#   >>> x = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + x, (x, x))
#   (tensor([2.]), tensor([2.]))
#
# In other words, the gradient is double-counted.  Deduplicating the arguments
# gives you an appropriate gradient:
#
#   >>> y = torch.randn(1, requires_grad=True)
#   >>> torch.autograd.grad(x + y, (x, y))
#   (tensor([1.]), tensor([1.]))
#
# HOW TO DEDUPLICATE:
#
# There are a few strategies, in order of preference:
#
# 1. For every duplicate argument to the function, detach it into
#    a separate leaf tensor, so that it is no longer duplicated.
#
#       PRO: The resulting compiled graph works for any configuration
#       of duplicated arguments.
#
#       CON: It does not (naively) work if you mutate the metadata of inputs:
#
#           def f(x, y):
#               x.transpose_(0, 1)
#               y.transpose_(0, 2)
#
#           x = torch.randn(2, 3, 4)
#           f(x, x)
#
#       The ordering of the transposes inside f dictates whether or not
#       you get [4, 2, 3] or [3, 4, 2].  This means that you cannot precompute
#       what metadata mutations should get applied to each input; you need to
#       assume they aren't duplicates (what we do today) or preserve
#       the original metadata mutations exactly in order, so that they work
#       for any duplicate configuration.
#
#       CON: It does not (naively) work if you mutate the data of inputs.
#       In particular, leaf tensors that require grad cannot be mutated,
#       this makes it impossible to differentiate with respect to the original
#       base.
#
# 2. For every duplicate argument to the function, remove it, so it is
#    no longer part of the "true" signature:
#
#       PRO: Implemented naively, it still works for metadata/data mutation.
#
#       CON: The resulting compiled graph is duplicate-specialized: it only
#       works if future calls duplicate arguments in exactly the same way.
#       Horribly, Dynamo doesn't guard on this at the moment.  But even if
#       it did, you could still end up recompiling a bunch of each duplicate.
#
# Our strategy is to do (1) if we can, and do (2) otherwise, erroring if
# Dynamo's guards are not enough.  In practice, this seems to cover
# everything.
#
def aot_wrapper_dedupe(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    compiler_fn,
    fw_metadata,
):
    # Use information about whether or not flat_fn mutates its arguments
    # or not to handle dupe args

    # Strategy 1: For any input that is not mutated, we can leafify it if we
    # need to remove a duplicate.
    leaf_flat_args = []
    args_set = set()
    ok = True

    for i, a in enumerate(flat_args):
        if not isinstance(a, torch.Tensor):
            leaf_flat_args.append(a)
        elif a not in args_set:
            args_set.add(a)
            leaf_flat_args.append(a)
        elif not fw_metadata.input_info[i].mutates_data and not fw_metadata.input_info[i].mutates_metadata:
            leaf_flat_args.append(a.detach().requires_grad_(a.requires_grad))
        else:
            ok = False
            break

    if ok:
        return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)

    if requires_subclass_dispatch(leaf_flat_args, fw_metadata):
        raise RuntimeError("""\
Encountered duplicate inputs that are mutated in the graph, but at least one input/output
to the graph is a tensor subclass. This is not supported today. You can try to
remove the aliasing yourself as a workaround, or otherwise file an issue on github.""")

    # export path: ban duplicate inputs for now, add later if requested.
    if aot_config.is_export:
        raise RuntimeError(f"""\
Encountered duplicated inputs that are mutated in the graph you are trying to export.
This functionality is currently not supported. If needed, please file a github issue.

fw_metadata={str(fw_metadata)}
        """)

    # Strategy 2: Duplicate specialize.
    #
    # In Haskell types, suppose you have:
    #
    #   add_dupe_args :: DedupedArgs -> Args
    #   remove_dupe_args :: Args -> DedupedArgs
    #
    #   compiler_fn
    #       :: (DedupedArgs -> R) -> DedupedArgs -> AOTConfig -> (DedupedArgs -> R)
    #   deped_compiler_fn
    #       :: (Args -> R) -> Args -> AOTConfig -> (Args -> R)
    #
    # Then the code below can be written in point-free style as:
    #
    #   deduped_compiler_fn f a c =
    #       compiler_fn (f . add_dupe_args) (remove_dupe_args a) c . remove_dupe_args
    #
    # Suppose you have:
    #
    #   [a, b, a, c]
    #
    # We want:
    #
    #   remove_dupe_args([a, b, a, c]) == [a, b, c]
    #   add_dupe_args([a, b, c]) == [a, b, a, c]
    #
    # This is done via (respectively):
    #
    #   seen_args = {a: 0, b: 1, c: 2}
    #   enumerate(add_dupe_map) = [  # how to get args from the deduped list
    #       (0, 0),
    #       (1, 1),
    #       (2, 0),
    #       (3, 2),
    #   ]
    #   keep_arg_mask = [True, True, False, True]

    seen_args = {}
    keep_arg_mask = []
    # Implicitly map duped arg position (list index) to de-duped arg position
    add_dupe_map: List[int] = []
    duped_arg_len = len(flat_args)

    j = 0  # index into deduped_flat_args
    for t in flat_args:
        if isinstance(t, torch.Tensor):
            if t in seen_args:
                keep_arg_mask.append(False)
                add_dupe_map.append(seen_args[t])
                continue
            seen_args[t] = j

        keep_arg_mask.append(True)
        add_dupe_map.append(j)
        j += 1
    assert len(add_dupe_map) == duped_arg_len, (
        f"Expects add_dupe_map to have length {duped_arg_len} but got {len(add_dupe_map)}"
    )

    # NB: Hot path, avoid set lookups here
    # TODO: Can avoid the zip here too, probably
    def remove_dupe_args(args):
        return [t for t, keep in zip(args, keep_arg_mask) if keep]

    def add_dupe_args(args):
        return [args[add_dupe_map[i]] for i in range(duped_arg_len)]

    deduped_flat_args = remove_dupe_args(flat_args)

    # Update our input metadata to remove duped input metadata.
    updated_fw_metadata = remove_dupe_metadata(fw_metadata, keep_arg_mask, add_dupe_map)

    tracing_context = TracingContext.get()
    if tracing_context and aot_config.aot_autograd_arg_pos_to_source:
        # TODO(voz): This structure is 1:1, we could consider an alternate structure like
        # kept_pos:[dupe_arg_pos], however, add_dupe_map is 1:1 so we would need a new structure there,
        # which feels like needless complexity for a tiny bit of efficiency at this point.
        for dupe_arg_pos, (kept_pos, keep_arg) in enumerate(zip(add_dupe_map, keep_arg_mask)):
            if not keep_arg:
                dupe_arg_source = aot_config.aot_autograd_arg_pos_to_source[dupe_arg_pos]
                kept_arg_source = aot_config.aot_autograd_arg_pos_to_source[kept_pos]
                tracing_context.guards_context.aotautograd_guards.append(DuplicateInputs(kept_arg_source, dupe_arg_source))

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        return flat_fn(*add_dupe_args(args))

    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
            wrapped_flat_fn,
            keep_input_mutations=fw_metadata.keep_input_mutations,
            is_train=fw_metadata.is_train,
        )(*deduped_flat_args)
        assert ref_fw_metadata == updated_fw_metadata, \
            f'ref_metadata={str(ref_fw_metadata)}, actual_metadata={str(updated_fw_metadata)}'

    compiled_fn = compiler_fn(wrapped_flat_fn, deduped_flat_args, aot_config, fw_metadata=updated_fw_metadata)

    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args):
        deduped_args = remove_dupe_args(args)
        args.clear()
        return compiled_fn(deduped_args)

    wrapped_compiled_fn._boxed_call = True

    # This can be uncommented when we properly guard for duplicates,
    # but right now we must not do it.
    # if not config.debug_assert:
    #     return wrapped_compiled_fn

    @wraps(wrapped_compiled_fn)
    def debugged_compiled_fn(args):
        # Test that the computed remove/add arg functions are an inverse
        new_args = add_dupe_args(remove_dupe_args(args))
        seen = {}
        for i, (x, y) in enumerate(zip(new_args, args)):
            seen[y] = None
            assert x is y, format_guard_bug_msg(
                aot_config,
                f"{describe_input(i, aot_config)} would be a duplicate of "
                f"{describe_input(add_dupe_map[i], aot_config)}",
            )
        # This is only an error if there is metadata mutation on both of
        # the duped arguments; in this case, we need to know what order
        # the metadata mutation applies in.  You'll get the correct result
        # otherwise, because a graph that assumes distinct inputs works if
        # you dupe the inputs (the gradient contributions from each input
        # will get summed up appropriately.)
        #
        # TODO: work out how to setup this assert correctly
        """
        assert len(seen) == unique_args, format_guard_bug_msg(aot_config,
            f"there would be {unique_args} distinct arguments"
        )
        """
        return wrapped_compiled_fn(args)

    debugged_compiled_fn._boxed_call = True

    return debugged_compiled_fn

# This layer handles the situation where you have two inputs that alias each other,
# and one of the inputs is mutated.
# We need to take special care to ensure that the mutation is applied to the other aliases in the graph.
#
# pre-condition: aot_wrapper_dedup has already run.
# (This function will in theory work if there are duplicate args.
# However, the synthetic base code path is a bit sub-optimal, and running with dupe'd inputs
# would cause us to hit that path more frequently).
def aot_wrapper_synthetic_base(
    flat_fn,
    flat_args: List[Tensor],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
    # Currently, the only reason we need to plumb this bool is because
    # the synthetic base code prohibits more cases in the autograd case than the inference case.
    needs_autograd: bool,
    compiler_fn,
):
    is_inference = not needs_autograd
    flat_args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
        flat_args, fw_metadata.input_info, is_inference=is_inference,
    )
    # Happy path: we don't need synthetic bases
    if synthetic_base_info is None:
        return compiler_fn(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)

    # export path: ban synthetic bases for now, add later if requested.
    if requires_subclass_dispatch(flat_args, fw_metadata):
        raise RuntimeError("""\
Encountered aliased inputs that are mutated in the graph, but at least one input/output
to the graph is a tensor subclass. This is not supported today. You can try to
remove the aliasing yourself as a workaround, or otherwise file an issue on github.""")

    if aot_config.is_export:
        raise RuntimeError(f"""\
Encountered aliased inputs that are mutated in the graph you are trying to export.
This functionality is currently not supported. If needed, please file a github issue.

synthetic_base_info={str(synthetic_base_info)}

fw_metadata={str(fw_metadata)}
        """)

    assert len(fw_metadata.input_info) == len(synthetic_base_info)

    # Update our forward metadata to take synthetic bases into account
    fw_metadata_updated, aliased_arg_idx_with_metadata_mutations = \
        create_synthetic_base_metadata(fw_metadata, synthetic_base_info, flat_args, flat_args_with_synthetic_bases)

    num_aliased_args_with_metadata_mutations = len(aliased_arg_idx_with_metadata_mutations)

    def unpack_synthetic_bases(primals: List[Any]) -> List[Any]:
        f_args_inner = []
        for inner_idx_or_tuple in synthetic_base_info:
            if isinstance(inner_idx_or_tuple, int):
                f_args_inner.append(primals[inner_idx_or_tuple])
            else:
                inner_base_idx, view_tensor = inner_idx_or_tuple
                base = primals[inner_base_idx]
                view_arg = gen_alias_from_base(
                    base, view_tensor, view_tensor.requires_grad
                )
                f_args_inner.append(view_arg)
        return f_args_inner

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        unpacked_args = unpack_synthetic_bases(args)
        # This is a bit subtle. The goal of this entire function (aot_dispatch_synthetic_bases)
        # is to relieve the downstream logic from having to reason about mutations on inputs that alias
        # each other, by replacing aliased inputs with a synthetic base.
        # One area where this breaks down a bit however is if one of those aliased inputs
        # experienced a metadata mutation.
        # We are now obligated to reapply the metadata mutation directly to the user's input;
        # it isn't enough to apply mutations back to the synthetic base in the downstream logic.
        #
        # The way we handle this is by pretending that those aliased inputs that experience metadata mutations
        # are additional outputs in the user's forward function.
        # The downstream logic will just treat these as "user outputs that alias inputs".
        # However, we will manually grab them at runtime here, use them to reapply the metadata mutation
        # to the user inputs, and not return them to the user.
        aliased_args_with_metadata_mutations = [
            x for i, x in enumerate(unpacked_args) if i in aliased_arg_idx_with_metadata_mutations]
        if len(aliased_args_with_metadata_mutations) > 0:
            return *(flat_fn(*unpacked_args)), *aliased_args_with_metadata_mutations
        else:
            return flat_fn(*unpacked_args)

    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(
            wrapped_flat_fn,
            keep_input_mutations=fw_metadata.keep_input_mutations,
            is_train=fw_metadata.is_train,
        )(*flat_args_with_synthetic_bases)
        assert ref_fw_metadata == fw_metadata_updated, (
            f'ref_metadata={pprint.pformat(partial_asdict(ref_fw_metadata))}, '
            f'\nactual_metadata={pprint.pformat(partial_asdict(fw_metadata_updated))}'
        )

    compiled_fn = compiler_fn(wrapped_flat_fn, flat_args_with_synthetic_bases, aot_config, fw_metadata=fw_metadata_updated)

    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args):
        args_with_synthetic_bases, synthetic_base_info = merge_view_inputs(
            args, fw_metadata.input_info, is_inference=is_inference
        )
        assert synthetic_base_info is not None
        aliased_args_w_metadata_mutations = [args[i] for i in aliased_arg_idx_with_metadata_mutations]
        args.clear()
        outs = compiled_fn(args_with_synthetic_bases)
        if num_aliased_args_with_metadata_mutations > 0:
            # This code does not handle **all** input metadata mutations.
            # Instead, it only handles metadata mutations on inputs that were converted into synthetic bases
            # (which only happens if at least one aliased input experienced a data mutation).
            # e.g:
            # def f(a, b):
            #     a.mul_(2)
            #     b.t_(1, 0)
            # f(x.view(2, 2), x.view(2, 2))
            mutated_metadata_inps = outs[-num_aliased_args_with_metadata_mutations:]
            user_outs = outs[:-num_aliased_args_with_metadata_mutations]
            for inp, mutated_inp in zip(aliased_args_w_metadata_mutations, mutated_metadata_inps):
                inp.as_strided_(mutated_inp.size(), mutated_inp.stride(), mutated_inp.storage_offset())
            return user_outs
        return outs

    return wrapped_compiled_fn


def describe_input(i, aot_config):
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        return f"input {i - aot_config.num_params_buffers}"

# The wrapper created by this function handles all of the runtime aliasing and mutation "epilogue" logic
# that needs to run after the compiled function.
#
# This function accepts a trace_joint flag, indicating whether or not we're generating the runtime
# epilogue for a forward-only inference graph, or for an autograd.Function.apply function.
# This is because there are some minor differences in how we treat these cases at runtime:
# - resize_() is currently handled in the inference case, but not fully handled in the autograd case.
# - the autograd cases inserts TensorAlias wrapper objects for outputs that alias inputs
def create_runtime_wrapper(
    compiled_fn,
    *,
    runtime_metadata: ViewAndMutationMeta,
    indices_of_inps_to_detach: List[int],
    trace_joint: bool,
    keep_input_mutations: bool,
    disable_amp: bool
):
    if not hasattr(compiled_fn, "_boxed_call"):
        compiled_fn = make_boxed_func(compiled_fn)

    def runtime_wrapper(*args):
        if trace_joint:
            args_ = list(args)
            # See Note [Detaching inputs that never need gradients]
            for idx in indices_of_inps_to_detach:
                if isinstance(args_[idx], torch.Tensor):
                    args_[idx] = args_[idx].detach()
            with torch.autograd._force_original_view_tracking(True):
                all_outs = call_func_with_args(
                    compiled_fn,
                    args_,
                    disable_amp=disable_amp,
                )
        else:
            # When we have an inference graph, we run with torch.no_grad.
            # It's possible to get an inference graph with inputs that require grad,
            # in which case we want to make sure autograd is disabled
            # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
            with torch.no_grad():
                all_outs = call_func_with_args(
                    compiled_fn,
                    args,
                    disable_amp=disable_amp,
                )

        num_mutated_inps = runtime_metadata.num_mutated_inputs
        num_metadata_mutated_inps = runtime_metadata.num_mutated_metadata_inputs
        num_intermediate_bases = runtime_metadata.num_intermediate_bases

        if keep_input_mutations:
            assert (
                len(all_outs)
                == num_metadata_mutated_inps + runtime_metadata.num_outputs + num_intermediate_bases
            )
            assert (
                len(runtime_metadata.mutated_inp_runtime_indices) == num_metadata_mutated_inps
            )
        else:
            assert (
                len(all_outs)
                == num_mutated_inps + runtime_metadata.num_outputs + num_intermediate_bases
            )
            assert (
                len(runtime_metadata.mutated_inp_runtime_indices) == num_mutated_inps
            )
        # Step 3: After running the compiled fw, apply updates to mutated inputs
        num_mutations_to_apply = len(runtime_metadata.mutated_inp_runtime_indices)
        if num_mutations_to_apply > 0:
            updated_inputs = all_outs[: num_mutations_to_apply]
            fw_outs = all_outs[num_mutations_to_apply :]

            for i, inpt_idx in enumerate(
                runtime_metadata.mutated_inp_runtime_indices
            ):
                meta = runtime_metadata.input_info[inpt_idx]
                if not meta.mutates_data and not meta.mutates_metadata:
                    continue
                original_inpt = args[inpt_idx]
                updated_inpt = updated_inputs[i]
                if meta.mutates_storage_metadata:
                    # mutates_storage_metadata means our input saw a x.set_(y) call.
                    # What if x **also** saw a data and/or a metadata mutation?
                    # (1) If the [meta]data mutation occurred after the set_(),
                    #     then there is no need to copy_() the data.
                    #     When we perform x.set_(x_updated), we are guaranteed that
                    #     x_updated already has the final version of the data/metadata
                    # (2) If a data mutation occurred before the set_().
                    #     This case seems very difficult to support.
                    #     TODO: discuss on the PR and decide if we want to tr to
                    #     either support it, or detect and ban it.
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    try:
                        original_inpt.set_(updated_inpt)
                    except RuntimeError:
                        # This happens when the sizes of the two tensors do not match, probably
                        # because we saw .data
                        # while we could call
                        # original_inpt.data = updated_inpt
                        # That's a little gross, let's use the "safe" pattern instead
                        with torch.no_grad():
                            version_counter = original_inpt._version
                            original_inpt.set_(updated_inpt)
                            if version_counter > 0:
                                version_counter = version_counter - 1
                            torch._C._autograd._unsafe_set_version_counter(original_inpt, version_counter)
                    continue
                if meta.mutates_metadata and not meta.mutates_data:
                    if trace_joint:
                        assert isinstance(updated_inpt, TensorAlias)
                        updated_inpt = updated_inpt.alias
                    # We need to grab the size/stride/storage_offset from the compiled forward,
                    # and use that to mutate the metadata of the input
                    original_inpt.as_strided_(
                        updated_inpt.size(),
                        updated_inpt.stride(),
                        updated_inpt.storage_offset(),
                    )
                else:
                    if meta.mutates_data and meta.mutates_metadata:
                        original_inpt.as_strided_(
                            updated_inpt.size(),
                            updated_inpt.stride(),
                            updated_inpt.storage_offset(),
                        )
                    else:
                        assert meta.mutates_data
                    if meta.is_leaf and original_inpt.requires_grad:
                        # We can hit this situation in this case:
                        #   def f(x):
                        #       x.detach().mul_(2)
                        #       return x + 1
                        # AOTAutograd will see a mutation in the above case, and try to
                        # apply a copy_() here, in the epilogue.
                        # But if x required gradients, and is a leaf, then autograd
                        # will yell at us for trying to mutate it.
                        # However, it's only possible to end up in this scenario (like the above)
                        # if all of the mutations to the leaf input were non-autograd-tracking mutations
                        # (aka mutations under no_grad(), or on detached views).
                        # In that case, we fully want to hide the mutation from autograd, so detaching is ok.
                        original_inpt.detach().copy_(updated_inpt)
                    else:
                        original_inpt.copy_(updated_inpt)
        else:
            fw_outs = all_outs

        # Step 4: Manually regenerate any outputs that are aliased to inputs, instead of
        # compiling them.
        if runtime_metadata.num_outputs_aliased > 0:
            # The compiled forward also returned intermediate bases. We don't want to return them to the user.
            if runtime_metadata.num_intermediate_bases > 0:
                fw_outs_no_intermediate_bases = fw_outs[
                    : -runtime_metadata.num_intermediate_bases
                ]
                intermediate_bases = fw_outs[-runtime_metadata.num_intermediate_bases:]
            else:
                fw_outs_no_intermediate_bases = fw_outs
                intermediate_bases = []
            assert len(fw_outs_no_intermediate_bases) == len(runtime_metadata.output_info)

            fw_outs_including_aliases = []
            for i, (o, info) in enumerate(zip(
                fw_outs_no_intermediate_bases, runtime_metadata.output_info
            )):
                if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]:
                    fw_outs_including_aliases.append(o)
                    continue
                if trace_joint:
                    assert isinstance(o, TensorAlias)
                    o_ = o.alias
                else:
                    o_ = o

                o_grad = runtime_metadata.output_info[i].requires_grad
                if info.output_type == OutputType.alias_of_input:
                    aliased_base_tensor = args[info.base_idx]
                    regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.is_input:
                    aliased_base_tensor = args[info.base_idx]
                    regenerated_out = aliased_base_tensor
                    fw_outs_including_aliases.append(regenerated_out)
                    continue
                elif info.output_type == OutputType.alias_of_intermediate:
                    base_tensor_list = intermediate_bases
                elif info.output_type == OutputType.alias_of_intermediate_save_as_output:
                    base_tensor_list = intermediate_bases
                else:
                    assert info.output_type == OutputType.alias_of_intermediate_base_is_user_output
                    base_tensor_list = fw_outs_no_intermediate_bases
                aliased_base_tensor = base_tensor_list[info.base_idx]
                # TODO: handle the custom autograd function case here.
                # We need a way to check whether a tensor came from a custom autograd fn from python,
                # AND a way to replay that custom view fn.
                regenerated_out = gen_alias_from_base(aliased_base_tensor, o_, o_grad)
                fw_outs_including_aliases.append(regenerated_out)
            ret_outs = fw_outs_including_aliases
        else:
            ret_outs = fw_outs

        if runtime_metadata.dynamic_outputs:
            for t, o in zip(ret_outs, runtime_metadata.output_info):
                if o.dynamic_dims is None:
                    continue
                if hasattr(t, '_dynamo_weak_dynamic_indices'):
                    t._dynamo_weak_dynamic_indices |= o.dynamic_dims
                else:
                    t._dynamo_weak_dynamic_indices = o.dynamic_dims.copy()
        if runtime_metadata.grad_enabled_mutation is not None:
            torch.set_grad_enabled(runtime_metadata.grad_enabled_mutation)
        return ret_outs
    return runtime_wrapper

# Calling convention: If we are running functionalized RNG, then outs consists
# of (user_outs, rng_offset)
def functionalized_rng_runtime_epilogue(metadata, outs, return_new_outs=True):
    if metadata.is_rng_op_functionalized:
        assert metadata.num_outputs_rng_offset == 1
        new_rng_offset = outs[-1]
        CUDARngStateHelper.set_new_offset(new_rng_offset)
        if return_new_outs:
            user_outs = outs[:-1]
            return user_outs
        else:
            return None
    return outs


def create_functionalized_rng_ops_wrapper(func, args, trace_joint=True):
    # Functionalization of rng ops changes the calling convention of the joint graph.
    # It goes from (primals, tangents) to (seed, offset, primals, tangents)
    # At runtime, we pass on the current seed and offset. This is hidden from
    # the user.
    fake_mode = detect_fake_mode()
    if fake_mode is None:
        fake_mode = nullcontext()

    def override_get_rng_state(device: Union[int, str, torch.device] = 'cuda'):
        out = PhiloxStateTracker.get_state_as_tensor()
        return out

    def override_set_rng_state(x, device: Union[int, str, torch.device] = 'cuda'):
        PhiloxStateTracker.set_state_from_tensor(x)

    def append_rng_offsets(args):
        if trace_joint:
            # args signature before: Tuple(fwd_outputs), Tuple(bwd_outputs)
            # args signature after: Tuple(fwd_outputs, new_fwd_rng_offset), Tuple(bwd_offset, new_bwd_rng_offset)
            return ((*args[0], PhiloxStateTracker.get_updated_fwd_offset()),
                    (*args[1], PhiloxStateTracker.get_updated_bwd_offset()))
        else:
            # args signature before: Tuple(fwd_outputs)
            # args signature after: Tuple(fwd_outputs, new_fwd_rng_offset)
            return (*args, PhiloxStateTracker.get_updated_fwd_offset())


    def traced_joint(primals, tangents, fwd_seed, fwd_base_offset, bwd_seed, bwd_base_offset):
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch("torch.cuda.set_rng_state", override_set_rng_state):
            return append_rng_offsets(func(primals, tangents))

    def traced_forward(*primals_fwd_seed_fwd_base_offset):
        # The signature is (*primals, seed, offset)
        with patch("torch.cuda.get_rng_state", override_get_rng_state), patch("torch.cuda.set_rng_state", override_set_rng_state):
            return append_rng_offsets(func(*primals_fwd_seed_fwd_base_offset[:-2]))

    if trace_joint:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
        bwd_seed, bwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        PhiloxStateTracker.record_state(bwd_seed, bwd_base_offset, "backward")
        return traced_joint, (*args, fwd_seed, fwd_base_offset, bwd_seed, bwd_base_offset)
    else:
        # Get the current seed and offset to setup tracing.
        fwd_seed, fwd_base_offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
        PhiloxStateTracker.record_state(fwd_seed, fwd_base_offset, "forward")
        return traced_forward, (*args, fwd_seed, fwd_base_offset)


# Output structure:
# - List[Tensor] if tracing an inference graph
# - Tuple[List[Tensor], List[Tensor]] if tracing a joint graph.
# This function effectively concats each inner list of subclass tensors
# into a (potentially longer) list of inner tensors.
#
# This function takes in a pytree of arguments and unwraps any tensor subclasses.
# Annoyingly, we can't use pytrees to perform the unwrapping, because unwrapping returns
# a list of tensors that we would then need to concat together.
# Instead, we specialize the logic for the inference vs. joint graph case.
# NOTE: this function is hot, since we unwrap tensor subclass inputs at runtime
def unwrap_tensor_subclasses(wrapped_args, *, is_joint_structure: bool):
    def concat_inner_tensors_from_subclasses(xs):
        xs_inner = []
        for x in xs:
            if isinstance(x, torch.Tensor) and is_traceable_wrapper_subclass(x):
                attrs, _ = x.__tensor_flatten__()
                xs_inner += [getattr(x, attr) for attr in attrs]
            else:
                xs_inner += [x]
        return xs_inner

    if is_joint_structure:
        assert isinstance(wrapped_args, tuple) and len(wrapped_args) == 2
        assert isinstance(wrapped_args[0], (tuple, list)) and isinstance(wrapped_args[1], (tuple, list))
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args[0])
        unwrapped_args_tangents = concat_inner_tensors_from_subclasses(wrapped_args[1])
        unwrapped_args = (unwrapped_args_fw, unwrapped_args_tangents)
    else:
        assert isinstance(wrapped_args, (list, tuple))
        unwrapped_args_fw = concat_inner_tensors_from_subclasses(wrapped_args)
        unwrapped_args = unwrapped_args_fw
    return unwrapped_args

# Turns a flattened list of tensor arguments into (maybe) subclass tensors.
# This function is used both at trace time and runtime, so we have an is_runtime flag telling us which context we're in.
def wrap_tensor_subclasses(
    unwrapped_args: List[Any],
    *,
    subclass_metas: List[Union[int, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int] = None,
    is_runtime: bool = False,
) -> List[Any]:
    wrapped_args = []
    num_args_tallied = 0
    for subclass_meta in subclass_metas:
        if isinstance(subclass_meta, int):
            wrapped_args.append(unwrapped_args[subclass_meta])
            num_args_tallied += 1
        else:
            assert isinstance(subclass_meta, SubclassCreationMeta)
            wrapped_args.append(subclass_meta.creation_fn(unwrapped_args, is_runtime=is_runtime))
            num_args_tallied += subclass_meta.arg_count

    # Note: [Partitioner handling for Subclasses, Part 2]
    # At the beginning of AOTAutograd, we collect metadata on the inputs and outputs of the user fw,
    # to figure out which inputs/outputs are subclasses, and how to reconstruct the subclasses after flattening them.
    #
    # When this function is called at runtime in the forward,
    # we have been passed a list of (flattened) dense-tensor fw-outs, and need to reconstruct any subclass fw outs.
    #
    # One reasonable question that you should ask: when should the dense_tensor -> subclass_tensor wrapping happen?
    # Answer: we do it **inside of our compiled autograd.Function**.
    # This seems like morally the right place: autograd happens above subclass desugaring,
    # so autograd should see actual tensor subclasses at runtime, and not flattened dense tensors.
    #
    # This causes a tricky interaction though: when we run the min-cut partitioner to divvy up the joint graph
    # into a forward and backward graph, we end up with some activations that show up as extra outputs
    # in the compiled forward graph, that are **not** user outputs.
    # These activations are not visible to the user, and so there's no need for us to wrap them back into subclasses.
    #
    # On top of that, when we first computed subclass metadata (in `run_functionalized_fw_and_collect_metadata`),
    # we computed subclass metadata on every forward output, but this did **not** include activations
    # created by the partitioner.
    # as a result, `unwrapped_args` here will correspond to (*unwrapped_user_fw_outs, *activations),
    # but `subclass_metas` will only correspond to subclass metatadata on `user_fw_outs`.
    # We then need to make sure that we return (*wrapped_user_fw_outs, *activations).
    if num_fw_outs_saved_for_bw is not None:
        assert len(unwrapped_args) == num_args_tallied + num_fw_outs_saved_for_bw
        activations = unwrapped_args[num_args_tallied:]
        if isinstance(wrapped_args, tuple) and isinstance(activations, tuple):
            return wrapped_args + activations
        return tuple(list(wrapped_args) + list(activations))
    else:
        assert len(unwrapped_args) == num_args_tallied
        return tuple(wrapped_args)

# Given a bunch of "dense" tensor arguments, this function (potentially) wraps them into tensor subclasses.
# This function carefully handles the inference vs. joint cases:
# - when is_joint_structure is True, args is (primals, tangents)
# - when is_joint_structure is False, args is [*primals]
def wrap_tensor_subclasses_maybe_joint(unwrapped_args, *, is_joint_structure: bool, meta: ViewAndMutationMeta) -> List[Any]:
    # Since this function is re-used for both inference and joint graphs,
    if is_joint_structure:
        assert isinstance(unwrapped_args, tuple) and len(unwrapped_args) == 2
        assert isinstance(unwrapped_args[0], (tuple, list)) and isinstance(unwrapped_args[1], (tuple, list))
        primals, tangents = unwrapped_args[0], unwrapped_args[1]
        wrapped_primals = wrap_tensor_subclasses(primals, subclass_metas=meta.subclass_inp_meta)
        wrapped_tangents = wrap_tensor_subclasses(tangents, subclass_metas=meta.subclass_tangent_meta)
        return (wrapped_primals, wrapped_tangents)
    else:
        wrapped_args = wrap_tensor_subclasses(unwrapped_args, subclass_metas=meta.subclass_inp_meta)
        return wrapped_args

# This wrapper handles the AOTDispatch runtime logic for tensor subclasses.
# At runtime, we have a compiled function that knows how to operate on the domain of DenseTensor -> DenseTensor,
# But the user might have passed us some tensor subclass inputs (or expect some subclass tensor outputs).
# This function handles the wrapping and unwrapping of tensor subclasses at runtime.
def aot_dispatch_subclass_wrapper(
    runtime_fn: Callable,
    *,
    subclass_metas: List[Union[int, SubclassCreationMeta]],
    num_fw_outs_saved_for_bw: Optional[int],
) -> Callable:
    def inner_fn(args):
        unwrapped_args = unwrap_tensor_subclasses(args, is_joint_structure=False)
        # expectation: runtime_fn is a boxed fn
        unwrapped_outs = runtime_fn(unwrapped_args)
        wrapped_outs = wrap_tensor_subclasses(
            unwrapped_outs, subclass_metas=subclass_metas, num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw, is_runtime=True)
        return wrapped_outs
    # box it
    inner_fn._boxed_call = True
    return inner_fn

def create_metadata_for_subclass(meta: ViewAndMutationMeta) -> ViewAndMutationMeta:
    # input infos
    input_info = []
    for inp, subclass_meta in zip(meta.input_info, meta.subclass_inp_meta):
        num_inps = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_inps):
            input_info.append(inp)

    # output infos
    output_info = []
    subclass_out_meta_user_outs_only = meta.subclass_fw_graph_out_meta[meta.num_mutated_data_inputs:]
    if meta.num_intermediate_bases > 0:
        subclass_out_meta_user_outs_only = subclass_out_meta_user_outs_only[:-meta.num_intermediate_bases]
    # sanity assert
    assert len(meta.output_info) == len(subclass_out_meta_user_outs_only)
    # Assume that the information on the output is shared by all of its inner tensors.
    for out, subclass_meta in zip(meta.output_info, subclass_out_meta_user_outs_only):
        num_outs = 1 if isinstance(subclass_meta, int) else subclass_meta.arg_count
        for _ in range(num_outs):
            output_info.append(out)

    # A bit hacky, but we don't actually care about all of the metadata here.
    # This metadata is used **underneath** both autograd and subclass de-sugaring,
    # So all we really care about is stuff like:
    # - num inputs/outputs (needed by the partitioner)
    # - input mutations (**not** used today, since we don't handle input mutations inside the subclass,
    #   although we should handle this eventually)
    #   TODO: add a test case to assert we error when this happens, instead of getting silent correctness
    num_intermediate_bases = None
    keep_input_mutations = meta.keep_input_mutations
    traced_tangents = None
    subclass_inp_meta = None
    subclass_fw_graph_out_meta = None
    subclass_tangent_meta = None

    metadata = ViewAndMutationMeta(
        input_info=input_info,
        output_info=output_info,
        num_intermediate_bases=num_intermediate_bases,
        keep_input_mutations=keep_input_mutations,
        traced_tangents=traced_tangents,
        subclass_inp_meta=subclass_inp_meta,
        subclass_fw_graph_out_meta=subclass_fw_graph_out_meta,
        subclass_tangent_meta=subclass_tangent_meta,
    )
    return metadata


SubclassTracingInfo = collections.namedtuple("SubclassTracingInfo", [
    "plain_tensor_trace_fn", "plain_tensor_args", "maybe_subclass_meta"
])

# Given a function operating on Subclass -> Subclass, returns an function that operates on Tensor -> Tensor
# Also returns:
# - the new set of arguments to pass into this function (now that tensor subclasses have been eliminated)
# - the updated ViewAndMutationMeta for this dense -> dense function.
# The other important arguments are:
# - flat_fn_maybe_joint: when is_joint_structure=True, this is the joint fw-bw function.
#                        when is_joint_structure=False, this is just the forward function.
# - fw_only: this is *always* the forward-only function.
#   Why do we need this? We need to collect updated ViewAndMutationMeta on our new dense -> dense functions.
#   In particular, we need this to tell the partitioner how many dense forward outputs there are.
def aot_dispatch_subclass(
    flat_fn_maybe_joint,
    args: List[Any],
    *,
    is_joint_structure: bool,
    meta: ViewAndMutationMeta,
    fw_only: Callable,
) -> "SubclassTracingInfo":
    # Skip logic if we don't need to trace through any subclasses
    req_subclass_dispatch = requires_subclass_dispatch(args, meta)
    if not req_subclass_dispatch:
        return SubclassTracingInfo(
            plain_tensor_trace_fn=flat_fn_maybe_joint,
            plain_tensor_args=args,
            maybe_subclass_meta=None,
        )

    # TODO: add subclass guards (later PR).

    # What's going on here? We need to compute subclass metadata about the outputs of the joint (grad_inputs).
    # Annoying: we don't know the grad input metas until we're in the middle of tracing the joint,
    # so we set it later, while we're tracing the joint (see inner_fn() below).
    # Another option would be to run our run_functionalized_fw_and_collect_metadata() function
    # directly on the joint, but this would hurt compile time (adding yet another pass through the joint).
    subclass_meta = SubclassMeta()

    def inner_fn(fn, args, *, use_trace_joint: bool):
        # Step 1: wrap tensor inputs into subclasses if necessary
        all_args = wrap_tensor_subclasses_maybe_joint(args, is_joint_structure=use_trace_joint, meta=meta)

        # Step 2: call the inner function, with our (maybe subclass) inputs
        wrapped_outs = fn(*all_args)

        if use_trace_joint:
            # See Note: [Computing Subclass Metadata about grad_inputs]
            # We also stash subclass info on our grad_inputs, if we're tracing the joint.
            nonlocal subclass_meta
            assert isinstance(wrapped_outs, tuple) and len(wrapped_outs) == 2
            # Don't need fw outs since we already have subclass metadata on them
            grad_inputs = wrapped_outs[1]
            subclass_meta.grad_input_metas = create_subclass_meta(grad_inputs)

        # Step 3: Unwrap any subclass outputs back into dense tensors
        unwrapped_outs = unwrap_tensor_subclasses(wrapped_outs, is_joint_structure=use_trace_joint)
        return unwrapped_outs

    def joint_fn(primals, tangents):
        return inner_fn(flat_fn_maybe_joint, (primals, tangents), use_trace_joint=True)

    def fw_fn(*primals):
        return inner_fn(flat_fn_maybe_joint, primals, use_trace_joint=False)

    def metadata_fn(*primals):
        return inner_fn(fw_only, primals, use_trace_joint=False)

    args_unwrapped = unwrap_tensor_subclasses(args, is_joint_structure=is_joint_structure)

    if is_joint_structure:
        primals_unwrapped = args_unwrapped[0]
        fn_to_trace = joint_fn
    else:
        primals_unwrapped = args_unwrapped
        fn_to_trace = fw_fn

    # Note: [Partitioner handling for Subclasses, Part 1]
    # The way the partitioner works is that:
    # (1) we pass is a single graph containing the joint fw/bw,
    #     where the # of graph outputs corresponds to # fw_outputs + # grad_inputs
    # (2) The partitioner accepts an arguments, num_fwd_outputs,
    #     and assumes that the first "num_fwd_outputs" graph outputs correspond
    #     to outputs of the forward graph.
    # How do tensor subclasses enter the picture?
    # the num_fwd_outputs in the final graph is actually non-trivial to compute,
    # because it can be influenced by input mutations and intermediate bases.
    # So we compute it by inspecting the current ViewAndMutationMeta object.
    # However, the original ViewAndMutationMeta that we computed was created
    # on the subclass -> subclass graph,
    # which can have a different number of outputs than the dense -> dense graph.
    # That's why we createa a fresh metadata object on the dense -> dense function here,
    # and plumb it back up to the partitioner.
    # See Note: [Partitioner handling for Subclasses, Part 2] for more info.
    meta_updated = run_functionalized_fw_and_collect_metadata(
        metadata_fn,
        keep_input_mutations=meta.keep_input_mutations,
        is_train=meta.is_train,
    )(*primals_unwrapped)

    subclass_meta.fw_metadata = meta_updated

    return SubclassTracingInfo(
        plain_tensor_trace_fn=fn_to_trace,
        plain_tensor_args=args_unwrapped,
        maybe_subclass_meta=subclass_meta,
    )


# Has the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_autograd_graph(flat_fn, flat_args: List[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    # traced_tangents corresponds to the set of outputs in the traced forward that should get grad_outputs in the traced backward.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs or intermediates, or any metadata-only input mutations.
    traced_tangents = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        fw_metadata.traced_tangents,
    )

    joint_inputs = (flat_args, traced_tangents)

    fn_prepared_for_autograd = fn_prepped_for_autograd(
        flat_fn,
        fw_metadata,
    )
    joint_fn_to_trace = create_joint(fn_prepared_for_autograd, aot_config=aot_config)

    joint_fn_to_trace, updated_joint_inputs = create_functionalized_fn(
        joint_fn_to_trace,
        joint_inputs,
        meta=fw_metadata,
        aot_config=aot_config,
        trace_joint=True,
    )

    subclass_tracing_info = aot_dispatch_subclass(
        joint_fn_to_trace, updated_joint_inputs, is_joint_structure=True, meta=fw_metadata, fw_only=flat_fn)

    joint_fn_to_trace = subclass_tracing_info.plain_tensor_trace_fn
    updated_joint_inputs = subclass_tracing_info.plain_tensor_args
    maybe_subclass_meta = subclass_tracing_info.maybe_subclass_meta

    fx_g = create_graph(joint_fn_to_trace, updated_joint_inputs, aot_config=aot_config)

    # There should be *NO* mutating ops in the graph at this point.
    assert_functional_graph(fx_g.graph)

    # Redundant with the check above, but worth having in case tracing introduced
    # a fake tensor. Unlikely.
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(fx_g)
    fx_g.graph.eliminate_dead_code()
    fx_g.recompile()
    # TODO: in AOTAutograd, we create metadata like _indices_of_inps_to_detach to detect
    # when we need to manually detach() some inputs in the forward.
    # Higher order ops might eventually need to do the same.
    if aot_config.is_export:
        assert maybe_subclass_meta is None, "aot_export_module does not support tensor subclass inputs for now."
        return fx_g
    return fx_g, updated_joint_inputs, maybe_subclass_meta

def aot_dispatch_autograd(flat_fn, flat_args: List[Any], aot_config: AOTConfig, *, fw_metadata: ViewAndMutationMeta):
    fx_g, joint_inputs, maybe_subclass_meta = aot_dispatch_autograd_graph(flat_fn, flat_args, aot_config, fw_metadata=fw_metadata)

    # Copied from aot_dispatch_autograd_graph.
    traced_tangents = pytree.tree_map(
        lambda x: x.detach().contiguous() if isinstance(x, Tensor) else x,
        fw_metadata.traced_tangents,
    )
    disable_amp = torch._C._is_any_autocast_enabled()

    if aot_config.enable_log:
        aot_joint_log.info("%s", lazy_format_graph_code("Joint graph", fx_g, aot_config.aot_id))

    with torch.no_grad():
        inner_meta = fw_metadata if maybe_subclass_meta is None else maybe_subclass_meta.fw_metadata
        with track_graph_compiling(aot_config, "joint"):
            # See Note: [Partitioner handling for Subclasses, Part 1]
            num_inner_fwd_outputs = (
                inner_meta.num_mutated_inputs
                + inner_meta.num_outputs
                + inner_meta.num_intermediate_bases
                + inner_meta.num_outputs_rng_offset
            )
            fw_module, bw_module = aot_config.partition_fn(
                fx_g, joint_inputs, num_fwd_outputs=num_inner_fwd_outputs
            )
            fw_outs = next(n for n in fw_module.graph.nodes if n.op == "output").args[0]
            # we only need to bookkeep the symints that are saved for bw, not any symints
            # the user forward might have returned in its own output
            fw_outs_saved_for_bw = fw_outs[num_inner_fwd_outputs:]
            num_fw_outs_saved_for_bw = len(fw_outs_saved_for_bw)
            symint_outs_saved_for_bw = [
                n for n in fw_outs_saved_for_bw if is_sym_node(n)
            ]
            fw_metadata.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            inner_meta.num_symints_saved_for_bw = len(symint_outs_saved_for_bw)
            _num_symints_saved_for_bw = len(symint_outs_saved_for_bw)

        # Note [Detaching inputs that never need gradients]
        # See https://github.com/pytorch/pytorch/issues/97745
        # Suppose we have a function like this that we want to compile:
        #
        # def f(x, y):
        #     return torch.mul(x, y.detach())
        #
        # What gradients should we compute for x and y?
        # By default, AOTAutograd will compute a gradient for **every** input that requires gradients,
        # and so we'll compute:
        #    x_grad_input = y
        #    y_grad_input = None
        # Does this preserve the semantics of eager mode?
        # Unfortunately, no.
        # Doing the above will cause autograd to **continue** to backprop the autograd tape
        # that was generated from constructing y.
        #
        # This is **different** from what would have happened in eager mode.
        # In eager mode, if we backprop through the output of this function, autograd will only traverse
        # the bit of the autograd tape corresponding to "x".
        # In particular, if a user had previously backpropped through y's autograd tape,
        # And then they try to backprop through the output of the above function,
        # then we'll hit the dreaded "Trying to backward through the graph a second time" error.
        #
        # You might think: If autograd sees that a gradient is None, shouldn't it stop early,
        # instead of continuing the backprop through the ancestors of that node in the graph?
        #
        # Autograd has two passes:
        # (1) a first pass that traverses the autograd graph and figures out which nodes need to be executed
        # (2) a second pass that actually goes ahead and executes each node when it becomes ready,
        #     propagating gradients
        # By the time we're executing a node and we see that it produces a None, the set of nodes to execute
        # is already locked-in.
        #
        # The fix: instead, we can recognize statically that the graph we're compiling will never contribute
        # gradients to y, and prevent autograd from trying to traverse y's autograd tape at all.
        # We can do this by manually detach'ing y before sending it through the `CompiledFunction`.
        #
        # Note that this solution is not bulletproof.
        # It's possible to construct a case where eager may or may not have have tried to autograd through y,
        # depending on the actual grad_outputs that were passed in during the backward.
        # There is no easy fix for this: the simplest fix would be to run with `retain_graph=True`,
        # allowing autograd to re-use the graph.
        #
        # An example of this case is:
        # def f(x):
        #     return x.detach() * 2, x * 3
        # If we were to only backprop through outs[0], in eager, we would stop
        # If we backward only on the first output, we shouldn't send a grad through x.
        # But the custom autograd function doesn't know that: it will materialize zero grads for x * 3
        # and we will end up with a zero grad at x.
        # If we later backprop through the second output, this will also require backprop'ing through x.
        # Meaning we'll need to use `retain_graph=True` to be able to backprop through x the second time.
        _indices_of_inps_to_detach = []
        bw_outs = next(n for n in bw_module.graph.nodes if n.op == "output").args[0]

        # TODO: we should apply the below "detach inputs if their gradients are statically known to be None"
        # optimization even if we have subclass inputs/outputs (we do not handle this today).
        # Computing which our our inputs get None gradients is a bit more complicated,
        # if any of our inputs are subclasses. Why?
        # (a) we need to make sure that we call .detach() on the input subclasses, since autograd sees subclasses.
        # (b) The grad_outputs that we AOT computed in our backward graph are the desugared tensor tensors,
        #     so we need to figure out which subclass fw inputs they map to.
        if maybe_subclass_meta is None:
            assert len(bw_outs) == len(fw_metadata.input_info) + inner_meta.num_outputs_rng_offset
            for i, (bw_out) in enumerate(bw_outs):
                if bw_out is None:
                    _indices_of_inps_to_detach.append(i)

        if aot_config.enable_log:
            aot_graphs_log.info("%s", lazy_format_graph_code("Forward graph", fw_module, aot_config.aot_id))
            aot_graphs_log.info("%s", lazy_format_graph_code("Backward graph", bw_module, aot_config.aot_id))

        with track_graph_compiling(aot_config, "forward"):
            # flat_args at this point might still be subclasses-
            # make sure to pass the unwrapped fake tensors into the compiler!
            adjusted_flat_args = joint_inputs[0]
            if config.functionalize_rng_ops:
                # Update example inputs for the fw_compiler
                fake_mode = detect_fake_mode()
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple(fake_mode)
                adjusted_flat_args.extend([seed, offset])
                # We are not clearing flat_args here because
                # 1) There is a check in the debug compiler at the end
                # 2) It does not matter as these are fake tensors

            if torch._guards.TracingContext.get():
                torch._guards.TracingContext.get().fw_metadata = inner_meta

            with TracingContext.report_output_strides() as fwd_output_strides:
                compiled_fw_func = aot_config.fw_compiler(
                    fw_module, adjusted_flat_args
                )
            if not hasattr(compiled_fw_func, "_boxed_call"):
                compiled_fw_func = make_boxed_func(compiled_fw_func)

            if maybe_subclass_meta is not None:
                # Why do we need to pass in num_fw_outs_saved_for_bw?
                # See Note: [Partitioner handling for Subclasses, Part 2]
                compiled_fw_func = aot_dispatch_subclass_wrapper(
                    compiled_fw_func,
                    subclass_metas=fw_metadata.subclass_fw_graph_out_meta,
                    num_fw_outs_saved_for_bw=num_fw_outs_saved_for_bw
                )
                if not hasattr(compiled_fw_func, "_boxed_call"):
                    compiled_fw_func = make_boxed_func(compiled_fw_func)

        # NB: It's important to compile backwards ahead of time, as this may
        # add extra guards which we need to apply to the Dynamo cache at
        # forwards
        with track_graph_compiling(aot_config, "backward"):
            placeholder_list = fx_placeholder_vals(bw_module)

            forward_saved_for_backwards_strides = None
            if fwd_output_strides is not None:
                forward_saved_for_backwards_strides = fwd_output_strides[inner_meta.tensors_saved_for_backwards_slice]

            # saved activations can have different stride to eager if
            # the compiler does layout optimization. We should restride the
            # tensor passed in for compiling the backward graph using the
            # saved tensor's stride.
            for i in range(len(placeholder_list)):
                ph_arg = placeholder_list[i]
                if not isinstance(ph_arg, torch.Tensor):
                    continue

                if forward_saved_for_backwards_strides is None:
                    continue

                real_stride = None
                # Per all_args calling convention
                j = i - len(symint_outs_saved_for_bw)
                if 0 <= j < len(forward_saved_for_backwards_strides):
                    real_stride = forward_saved_for_backwards_strides[j]
                if real_stride is None:
                    continue

                # Comparing ph_arg.stride() with real_stride directly may
                # cause dynamic dimensions in ph_arg being specialized to static
                # value. Using the hints to avoid that.
                if _get_hints(ph_arg.stride()) != real_stride:
                    # Note that here we use the stride of the real tensor to
                    # restride a FakeTensor. This does not cause trouble
                    # for dynamic shape since this code path only get
                    # executed if layout optimization is enabled. And we
                    # disable layout optimization for dynamic shape right
                    # now.
                    #
                    # A solution that decide stride order based on real
                    # tensor's stride and then apply that stride order to
                    # the FakeTensor does not work smoothly since some
                    # tensor's layout is not 'dense'. E.g. mixnet_l has a
                    # tensor with size [8, 64, 112, 112] and strides
                    # (2408448, 1, 21504, 192). The solution mentioned will
                    # decide a stride of (802816, 1, 7168, 64) for this
                    # tensor which is wrong.
                    placeholder_list[i] = ph_arg.as_strided(ph_arg.size(), real_stride)

            compiled_bw_func = None
            if len(symint_outs_saved_for_bw):
                context = torch._C._DisableAutocast if disable_amp else nullcontext
                with context():
                    try:
                        compiled_bw_func = aot_config.bw_compiler(
                            bw_module, placeholder_list
                        )
                    except Exception:
                        log.warning(
                            "failed to eagerly compile backwards for dynamic, suppressing in case backwards not needed",
                            exc_info=True
                        )

    saved_context = TracingContext.get()

    class CompiledFunction(torch.autograd.Function):
        compiled_fw = compiled_fw_func
        compiled_bw = compiled_bw_func
        metadata = fw_metadata
        maybe_subclass_metadata: Optional[SubclassMeta] = maybe_subclass_meta
        num_symints_saved_for_bw = _num_symints_saved_for_bw

        @staticmethod
        def _compiled_autograd_key(ctx):
            return (aot_config.aot_id, *ctx.symints)

        @staticmethod
        def forward(ctx, *deduped_flat_tensor_args):
            args = deduped_flat_tensor_args
            if CompiledFunction.metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                seed, offset = CUDARngStateHelper.get_torch_state_as_tuple()
                args = (*args, seed, offset)
            # There is a pretty complicated calling convention around what the compiled fw returns.
            # The full list of outputs and their relative order is:
            # (*mutated_inputs, *fw_outs, *fw_intermediate_bases, *saved_tensors, *saved_symints)
            # - Note that in the synthetic bases case, mutated_inputs will correspond to an updated version
            #   of the original view, and not the synthetic base
            fw_outs = call_func_with_args(
                CompiledFunction.compiled_fw,
                args,
                disable_amp=disable_amp,
            )

            num_outputs = CompiledFunction.metadata.num_outputs
            num_outputs_aliased = CompiledFunction.metadata.num_outputs_aliased
            num_intermediate_bases = CompiledFunction.metadata.num_intermediate_bases
            num_symints_saved_for_bw = CompiledFunction.num_symints_saved_for_bw
            num_mutated_inputs = CompiledFunction.metadata.num_mutated_inputs
            num_mutated_metadata_only_inputs = (
                CompiledFunction.metadata.num_mutated_metadata_only_inputs
            )
            num_forward_returns = CompiledFunction.metadata.num_forward_returns
            num_forward = CompiledFunction.metadata.num_forward

            # Partitioners must put symint arguments at the end separate from tensor arguments
            tensors_saved_for_backwards = fw_outs[
                CompiledFunction.metadata.tensors_saved_for_backwards_slice
            ]
            assert all(
                isinstance(x, torch.Tensor) for x in tensors_saved_for_backwards
            )
            # See Note [Detaching saved tensors in AOTAutograd]
            ctx.save_for_backward(*(x.detach() if x._is_view() else x for x in tensors_saved_for_backwards))
            symint_outs = fw_outs[CompiledFunction.metadata.symints_saved_for_backwards_slice]
            assert all(
                isinstance(x, (int, float, torch.SymInt, torch.SymFloat))
                for x in symint_outs
            ), str([type(x) for x in symint_outs])
            ctx.symints = symint_outs

            raw_returns = fw_outs[0:num_forward_returns]

            # Wrap all autograd.Function.forward() outputs that are aliases
            # so that autograd.Function doesn't treat them as tensors
            if num_mutated_metadata_only_inputs > 0:
                for i, idx in enumerate(
                    CompiledFunction.metadata.mutated_inp_indices
                ):
                    # We could make this faster by only looping over inputs with metadata-only mutations
                    # (instead of looping over inputs with either data or metadata mutations), but there shouldn't be many.
                    info = CompiledFunction.metadata.input_info[idx]
                    if info.mutates_metadata and not info.mutates_data:
                        raw_returns[i] = TensorAlias(raw_returns[i])

                if config.debug_assert:
                    user_mutated_inputs_raw = raw_returns[0:num_mutated_inputs]
                    mut_inp_infos = [
                        x for x in CompiledFunction.metadata.input_info if x.mutates_data or x.mutates_metadata
                    ]
                    assert len(user_mutated_inputs_raw) == len(mut_inp_infos)

            if CompiledFunction.metadata.num_unsafe_view_outputs > 0:
                for idx in CompiledFunction.metadata.unsafe_view_out_indices:
                    raw_return_idx = num_mutated_inputs + idx
                    o = raw_returns[raw_return_idx]
                    raw_returns[raw_return_idx] = torch.ops.aten._unsafe_view(o, o.shape)

            if num_outputs_aliased > 0:
                for idx in CompiledFunction.metadata.aliased_out_indices:
                    raw_return_idx = num_mutated_inputs + idx
                    raw_returns[raw_return_idx] = TensorAlias(raw_returns[raw_return_idx])

                if config.debug_assert:
                    intermediates_raw = raw_returns[num_mutated_inputs + num_outputs:]
                    assert not any(isinstance(x, TensorAlias) for x in intermediates_raw)

            # invariant: intermediate bases always require gradients, so we don't have to
            # consider marking them as non-differentiable.
            raw_returns_not_including_intermediate_bases = raw_returns[:num_mutated_inputs + num_outputs]

            raw_returns_meta = (
                [x for x in CompiledFunction.metadata.input_info if x.mutates_data or x.mutates_metadata]
                + CompiledFunction.metadata.output_info
            )

            fw_outs_not_requiring_grad = [
                x
                for (i, x) in enumerate(raw_returns_not_including_intermediate_bases)
                if isinstance(x, torch.Tensor)
                and not raw_returns_meta[i].requires_grad
            ]
            ctx.mark_non_differentiable(*fw_outs_not_requiring_grad)
            ctx._materialize_non_diff_grads = False

            functionalized_rng_runtime_epilogue(
                CompiledFunction.metadata,
                fw_outs[num_forward_returns:num_forward],
                return_new_outs=False
            )
            return tuple(raw_returns)

        @staticmethod
        def backward(ctx, *flat_args):
            # Calling convention: we expect a grad_out passed to the backward:
            # - for every output of the fw that does *not* alias an input or graph intermediate
            # - for every updated_input generated by the fw that does *not* alias an input (aka only data-mutations)
            # - for every graph intermediate that we need to use to generate an output later.
            # The other outputs in the autograd.Function.forward that do *not* show up in the backward include:
            # - outputs that alias inputs or graph intermediates
            # - updated inputs due to metadata-only mutations.
            # We need to return them in the forward, but ensure that they all do not get gradients in the backward,
            # and we filter them out here before passing the remaining grad_outputs into the compiled backward.
            num_mutated_inps = CompiledFunction.metadata.num_mutated_inputs
            num_intermediate_bases = CompiledFunction.metadata.num_intermediate_bases
            expected_grad_outs = (
                CompiledFunction.metadata.num_outputs + num_mutated_inps + num_intermediate_bases
            )

            assert len(flat_args) == expected_grad_outs
            out_info = CompiledFunction.metadata.output_info

            inp_tangents, out_tangents, intermediate_base_tangents = (
                flat_args[0:num_mutated_inps],
                flat_args[num_mutated_inps:num_mutated_inps + CompiledFunction.metadata.num_outputs],
                flat_args[num_mutated_inps + CompiledFunction.metadata.num_outputs:],
            )
            # input_info contains info on *every* input,
            # But in the backward(), we are only given grad outputs for every mutated input
            # We then need to filter out the grad outputs that correspond to metadata-only mutations or don't require grad
            mutated_inp_indices = CompiledFunction.metadata.mutated_inp_indices
            input_info = CompiledFunction.metadata.input_info
            assert len(inp_tangents) == len(mutated_inp_indices)
            inp_tangents_filtered = [
                x
                for x, info_idx in zip(inp_tangents, mutated_inp_indices)
                if input_info[info_idx].mutates_data and input_info[info_idx].requires_grad
            ]
            # We also need to filter out grad outputs that correspond to outputs aliasing inputs/intermediates
            out_tangents_filtered = [
                x
                for x, info in zip(out_tangents, out_info)
                if info.output_type in [OutputType.non_alias, OutputType.unsafe_view_alias, OutputType.custom_function_view]
                and issubclass(info.raw_type, torch.Tensor)
                and info.requires_grad
            ]
            # intermediate bases always require gradients, and always participate in the backward graph.
            flat_bw_args_with_grads = [*inp_tangents_filtered, *out_tangents_filtered, *intermediate_base_tangents]
            num_flat_bw_args_with_grads = len(flat_bw_args_with_grads)

            # sanity asserts
            # metadata_only_inps = [
            #     x for x, info_idx in zip(inp_tangents, mutated_inp_indices)
            #     if not input_info[info_idx].mutates_data
            # ]
            # aliased_outputs = [
            #     x for x, info in zip(out_tangents, out_info) if info.output_type != OutputType.non_alias]
            # assert all(x is None for x in metadata_only_inps)
            # assert all(x is None for x in aliased_outputs)

            rng_args = []
            if CompiledFunction.metadata.is_rng_op_functionalized:
                # Add the seed and offset to args
                rng_args = CUDARngStateHelper.get_torch_state_as_tuple()

            all_args = [
                *ctx.symints,
                *ctx.saved_tensors,
                *flat_bw_args_with_grads,
                *rng_args
            ]
            del flat_bw_args_with_grads

            tangents_start_idx = len(all_args) - num_flat_bw_args_with_grads - len(rng_args)
            tangents_end_idx = len(all_args) - len(rng_args)

            # Note: [AOTAutograd Backward Guards]
            # During AOTDispatch, we eagerly create and trace out a joint fw-bw graph.
            # Doing so requires us to "guess" about some of the metadata of our grad_outputs.
            #
            # In particular: if an output to the forward is a plain tensor or a subclass,
            # its corresponding grad_output in the backward **may or may not** be
            # a plain tensor or a subclass. The main cases are:
            # (1) If an output is a plain tensor, its grad_out will also be a plain tensor,
            #     *unless* the output is used in some subclass compute later in the forward graph,
            #     which will cause its grad_output to become a subclass
            # (2) If an output is a subclass, its grad_out will also be a subclass,
            #     *unless* the output of the forward did not actually participate in the gradient computation,
            #     in which case autograd will insert a plain tensor of zeros for the grad_output.
            #     We could avoid this case with `torch.autograd.Function.set_materialize_grads`,
            #     although this is not turned on today in AOTAutgrad and would require more work.
            #
            # Today, we make a guess on subclass-ness based on the above examples,
            # and hard-error in the backward if we guessed wrong.
            #
            # In the future, we should add backward guards that would allow us to
            # properly handle this case instead of erroring: we would need to retrace the backward graph,
            # since we might produce an entirely different trace if our grad_outputs are subclass or not.
            assert len(CompiledFunction.metadata.output_types) == num_flat_bw_args_with_grads
            grad_output_types = [type(x) for x in all_args[-num_flat_bw_args_with_grads:]]
            # In general, we can add more asserts/guards here for when we partitioned
            # with incorrect assumptions about the grad_outputs.
            # Normalize FakeTensor -> torch.Tensor
            # - during tracing our types are FakeTensor
            # - at runtime in the backward our types are torch.Tensor...
            # - unless we're running compiled backward, in which case they are also FakeTensor
            grad_output_types_ = [torch.Tensor if x is FakeTensor else x for x in grad_output_types]
            assert grad_output_types_ == CompiledFunction.metadata.output_types, f"""\
We incorrectly attempted to compile the backward with incorrect subclass metadata.
If you run into this error, please file an issue.
Expected grad_output types: {str(CompiledFunction.metadata.output_types)}
Got grad_output types: {str(grad_output_types)}"""

            # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
            if CompiledFunction.maybe_subclass_metadata is not None:
                # Get the number of tangents after unwrapping
                len_tangents = len(unwrap_tensor_subclasses(
                    all_args[tangents_start_idx: tangents_end_idx], is_joint_structure=False
                ))
                all_args = unwrap_tensor_subclasses(all_args, is_joint_structure=False)
                tangents_start_idx = len(all_args) - len_tangents - len(rng_args)
                tangents_end_idx = tangents_start_idx + len_tangents

            # Make the tangents contiguous. Note that we must do this after subclass desugaring
            # because inputs to inductor have to be contiguous
            all_args = [
                t.contiguous() if tangents_start_idx <= i < tangents_end_idx else t
                for i, t in enumerate(all_args)
            ]

            def call_compiled_backward():
                if ctx._is_compiled_autograd_tracing():
                    # For compiled autograd, run raw FX graph so that it can be inlined into the larger graph
                    symints = ctx._get_compiled_autograd_symints()
                    assert len(symints) == len(ctx.symints)
                    all_args[:len(symints)] = symints
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with context():
                        out = normalize_as_list(bw_module(*all_args))
                    out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
                    return tuple(out)
                ctx.maybe_clear_saved_tensors()
                if CompiledFunction.compiled_bw is None:
                    context = torch._C._DisableAutocast if disable_amp else nullcontext
                    with tracing(saved_context), context(), track_graph_compiling(aot_config, "backward"):
                        CompiledFunction.compiled_bw = aot_config.bw_compiler(
                            bw_module, placeholder_list
                        )

                out = call_func_with_args(
                    CompiledFunction.compiled_bw,
                    all_args,
                    steal_args=True,
                    disable_amp=disable_amp,
                )

                out = functionalized_rng_runtime_epilogue(CompiledFunction.metadata, out)
                return tuple(out)

            if torch.is_grad_enabled() and any(t.requires_grad for t in all_args if isinstance(t, torch.Tensor)):
                # Ensure that the graph is connected, and error if double backward is performed.
                # See comment for why once_differentiable is not sufficient:
                # https://github.com/pytorch/pytorch/pull/92348/files#r1072962107
                class CompiledFunctionBackward(torch.autograd.Function):
                    @staticmethod
                    def forward(ctx, *unused_args):
                        outs = call_compiled_backward()
                        # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
                        if CompiledFunction.maybe_subclass_metadata is not None:
                            outs_wrapped = wrap_tensor_subclasses(
                                outs, subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas)
                            return outs_wrapped
                        return outs

                    @staticmethod
                    def backward(ctx, *args):
                        raise RuntimeError("torch.compile with aot_autograd does not currently support double backward")

                CompiledFunctionBackward._compiled_autograd_key = CompiledFunction._compiled_autograd_key

                # Pass args even though they're unused, so that the graph is built
                out = CompiledFunctionBackward.apply(*all_args)
            else:
                out = call_compiled_backward()

            # TODO: figure out how to refactor the backward properly so I can use aot_dispatch_subclass_wrapper() here.
            if CompiledFunction.maybe_subclass_metadata is not None:
                outs_wrapped = wrap_tensor_subclasses(
                    out, subclass_metas=CompiledFunction.maybe_subclass_metadata.grad_input_metas)
                return outs_wrapped
            return out

    compiled_function = create_runtime_wrapper(
        CompiledFunction.apply,
        runtime_metadata=fw_metadata,
        indices_of_inps_to_detach=_indices_of_inps_to_detach,
        trace_joint=True,
        keep_input_mutations=False,
        disable_amp=disable_amp
    )

    if not config.debug_assert:
        return compiled_function

    flat_requires_grad = [
        a.requires_grad if isinstance(a, Tensor) else None for a in flat_args
    ]

    @wraps(compiled_function)
    def debug_compiled_function(*args):
        # TODO: Check aliasing relationships
        # TODO: Check strides for metadata mutation
        # (NB: ideally, this logic is factored out of this function and
        # you move these debug checks there)

        # Check requires grad.  Bad case is when we compiled with
        # requires_grad = False, but input requires_grad = True
        # (vice versa is OK; we compute a gradient and then throw
        # it away when it hits the input.)
        for i, a in enumerate(args):
            can_require_grad = flat_requires_grad[i]
            if can_require_grad is None:
                assert not isinstance(a, Tensor)
            elif not can_require_grad:
                assert not a.requires_grad, format_guard_bug_msg(
                    aot_config,
                    f"{describe_input(i, aot_config)} would not require grad",
                )

        return compiled_function(*args)

    return debug_compiled_function


@dynamo_timed
def create_aot_dispatcher_function(
    flat_fn, flat_args: List[Any], aot_config: AOTConfig
):
    """
    Traces the forward and backward graphs of the attr:`flat_fn` to generate a
    joint graph. The joint graph is an Fx graph with Aten ops. Please refer to
    the tracing mechanism to understand the graph capturing details.

    The joint graph is then passed through attr:`partition_fn` to isolate the
    forward and backward portions, which are then respectively compiled via the
    provided attr:`fw_compiler` and attr:`bw_compiler`.

    The resulting compiled forward and backward graphs are then wrapped up in a
    ``torch.autograd.Function`` object.

    The calling convention here is that the first aot_config.num_params_buffers
    inputs in flat_args are parameters and buffers, and the rest are inputs.

    We use this to assume that parameters/buffer's shapes don't change.

    Note: this function is used both by aot_function and aot_export (controlled by aot_config.is_export)
        When aot_config.is_export is True, we return an FX graph + metadata
        When aot_config.is_export is False, we return an ordinary runtime function
    """

    # This is the main entry point.
    # TODO: Chillee argues that dynamo itself should pass in fake tensors to
    # the list of arguments when compiling; at the moment we do not do this

    if aot_config.decompositions is None:
        aot_config.decompositions = {}


    aot_config.decompositions = {
        **aot_autograd_decompositions,
        **aot_config.decompositions,
    }

    if config.functionalize_rng_ops:
        # Update the decompositions with functionalized random decompositions
        aot_config.decompositions = {
            **rng_decompositions,
            **aot_config.decompositions,
        }

    # Check flat_args to see if they're already fake.  If so, use that fake
    # mode instead.

    fake_mode = detect_fake_mode(flat_args)
    if fake_mode is None:
        shape_env = ShapeEnv() if aot_config.dynamic_shapes else None
        fake_mode = FakeTensorMode(shape_env=shape_env)
    else:
        shape_env = fake_mode.shape_env

    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    with torch.autograd.set_multithreading_enabled(
        False
    ), preserve_rng_state(), fake_mode, python_dispatcher_mode, PhiloxStateTracker():

        def process_inputs(flat_args):
            def convert(idx, x):
                if shape_env is not None:
                    from torch._dynamo.source import ConstantSource
                    if isinstance(x, int):
                        source = ConstantSource(f"sym_{idx}")
                        return shape_env.create_symintnode(
                            shape_env.create_symbol(x, source),
                            hint=x,
                            source=source
                        )
                if not isinstance(x, torch.Tensor):
                    return x
                if isinstance(x, FakeTensor):
                    assert x.fake_mode is fake_mode
                    return x
                if is_traceable_wrapper_subclass(x):
                    attrs, _ = x.__tensor_flatten__()
                    if all(isinstance(getattr(x, attr), FakeTensor) for attr in attrs):
                        assert all(getattr(x, attr).fake_mode is fake_mode for attr in attrs)
                        return x
                static_shapes = idx < aot_config.num_params_buffers and config.static_weight_shapes
                widr = WeakIdRef(x)
                tc = torch._guards.TracingContext.get()
                # TODO(voz): Doc goes here
                if tc and widr in tc.weak_tensor_ref_to_fakification_policy:
                    policy = tc.weak_tensor_ref_to_fakification_policy[widr]
                    out = fake_mode.from_tensor(x, static_shapes=static_shapes, ignore_subclass=policy.ignore_subclass, dynamic_dims=policy.dynamic_dims, constraint_dims=policy.constraint_dims)
                else:
                    out = fake_mode.from_tensor(x, static_shapes=static_shapes)
                return out

            return [convert(idx, x) for idx, x in enumerate(flat_args)]

        fake_flat_args = process_inputs(flat_args)

        needs_autograd = (
            any(x.requires_grad for x in fake_flat_args if isinstance(x, Tensor))
            and torch.is_grad_enabled()
        )

        with enable_python_dispatcher():
            # Patch set_rng_state as set_rng_state with fake tensors is
            # nonsensical. This does not affect the collection of metadata.
            with patch("torch.cuda.set_rng_state", lambda *args: None):
                fw_metadata = run_functionalized_fw_and_collect_metadata(
                    flat_fn,
                    keep_input_mutations=aot_config.keep_inference_input_mutations and not needs_autograd,
                    is_train=needs_autograd,
                )(*fake_flat_args)

                req_subclass_dispatch = requires_subclass_dispatch(fake_flat_args, fw_metadata)

                if needs_autograd and not any(x.requires_grad for x in fw_metadata.output_info):
                    # We realized that none of the outputs require grad,
                    # so we actually have an inference graph.
                    needs_autograd = False
                    # A bit silly: right now in the subclass codepath, our ViewAndMutationMeta
                    # changes depending on whether we pass in is_train / keep_input_mutations,
                    # so we're forced to recompute the metadata.
                    # TODO: refactor the subclass path of run_functionalized_fw_and_collect_metadata
                    # so that this is unnecessary.
                    if req_subclass_dispatch:
                        fw_metadata = run_functionalized_fw_and_collect_metadata(
                            flat_fn,
                            keep_input_mutations=aot_config.keep_inference_input_mutations and not needs_autograd,
                            is_train=needs_autograd,
                        )(*fake_flat_args)
                    else:
                        fw_metadata = ViewAndMutationMeta(
                            input_info=fw_metadata.input_info,
                            output_info=fw_metadata.output_info,
                            num_intermediate_bases=fw_metadata.num_intermediate_bases,
                            keep_input_mutations=aot_config.keep_inference_input_mutations and not needs_autograd,
                            traced_tangents=fw_metadata.traced_tangents,
                            subclass_inp_meta=fw_metadata.subclass_inp_meta,
                            subclass_fw_graph_out_meta=fw_metadata.subclass_fw_graph_out_meta,
                            subclass_tangent_meta=fw_metadata.subclass_tangent_meta,
                            is_train=needs_autograd,
                        )


        if fw_metadata.num_intermediate_bases > 0:
            assert not req_subclass_dispatch, f"""\
torch.compile is currently being used with tensor subclass inputs:
{','.join([str(type(x)) for x in fake_flat_args])}. We are attempting to a compile a graph with two graph outputs
that alias one another, which is currently unsupported in the subclass use case. If you run into this,
please file a github issue"""

        if aot_config.is_export:
            # aot_export: ban input metadata mutations for now to keep shared code paths simpler.
            # Keeping .resize_() in the graph will require some work
            # Allowing it but keeping the graph functional will require some calling convention changes.
            if len([x for x in fw_metadata.input_info if x.mutates_metadata]) != 0:
                raise RuntimeError(f"""\
Found an input that received a metadata mutation, through e.g. a call to `.resize_()` or `.transpose_()`.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}""")
            # In export, banning data mutations on inputs that require grad for now.
            # This should be rare, and is tricky to get right. When we trace the backward,
            # we currently trace with autograd.grad instead of .backward(), which makes it difficult
            # to ensure that we run autograd all the way through the input **before** it saw the mutation.
            if len([x for x in fw_metadata.input_info if x.requires_grad and x.mutates_data]) != 0:
                raise RuntimeError(f"""\
Found a graph input that requires gradients, and received a mutation.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}""")
            if req_subclass_dispatch:
                raise RuntimeError("""\
aot_export is not currently supported with traceable tensor subclass.
If you need this feature, please comment on <CREATE_ISSUE_LINK>""")

            # Need to decide on a strategy for functionalized RNG: toggling via global config seems bad,
            # and turning it on will require a non-trivial calling convention change for any export runtime.
            if config.functionalize_rng_ops:
                raise RuntimeError("""\
Functionalized RNG is not currently supported in the aot_export workflow. Please file a github issue,
or otherwise set torch._functorch.config.functionalize_rng_ops = False.""")

        # crappy version of dispatcher
        # TODO: Do this properly
        if needs_autograd:
            # For now, aot_dispatch_autograd knows to explicitly return a graph
            # when run with export, and an opaque callable otherwise.
            # In theory we could factor these out, but I wanted to let the dust
            # settle on how functionalized rng fits into export first.
            compiler_fn = aot_dispatch_autograd_graph if aot_config.is_export else aot_dispatch_autograd
        else:
            # aot_dispatch_base_graph contains only the "graph bits", while aot_dispatch_base
            # includes some extra work around handling a runtime epilogue.
            compiler_fn = aot_dispatch_base_graph if aot_config.is_export else aot_dispatch_base

        compiler_fn = partial(aot_wrapper_synthetic_base, compiler_fn=compiler_fn, needs_autograd=needs_autograd)
        compiler_fn = partial(aot_wrapper_dedupe, compiler_fn=compiler_fn)
        # You can put more passes here

        compiled_fn = compiler_fn(flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata)
        if aot_config.is_export:

            mutated_user_inp_locs = [
                idx - aot_config.num_params_buffers
                for idx in fw_metadata.mutated_inp_indices
                if idx >= aot_config.num_params_buffers
            ]
            if len(mutated_user_inp_locs) > 0:
                raise RuntimeError(f"""
Found following user inputs located at {mutated_user_inp_locs} are mutated. This is currently banned in the aot_export workflow.
If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}""")

            # During export, we don't get back a callable - we get back the raw fx graph
            # (either a joint or an inference-only graph)
            assert isinstance(compiled_fn, torch.fx.GraphModule)
            return compiled_fn, fw_metadata

        if not hasattr(compiled_fn, "_boxed_call"):
            compiled_fn = make_boxed_func(compiled_fn)

        return compiled_fn


# Inspired by autodidax (thanks!)
class PytreeThunk:
    spec = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple = (
        None  # if the output spec is a tuple/list, we won't bother unflattening it.
    )
    is_really_simple = None  # if the output spec is a LeafSpec

    def set(self, spec):
        assert self.spec is None or self.spec == spec
        self.spec = spec
        if type(self.spec) in [tuple, list] and all(
            isinstance(i, pytree.LeafSpec) for i in spec.children_specs
        ):
            self.is_simple = True
        if isinstance(self.spec, pytree.LeafSpec):
            self.is_really_simple = True

    def unflatten(self, x):
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        return pytree.tree_unflatten(x, self.spec)


def create_functional_call(mod, params_spec, params_len):
    # Redundant with dynamo, but worth having in case this gets invoked elsewhere.
    # https://github.com/pytorch/pytorch/issues/103569

    def functional_call(*args, **kwargs):
        with stateless._reparametrize_module(
            mod, pytree.tree_unflatten(args[:params_len], params_spec)
        ):
            if isinstance(mod, torch.fx.GraphModule):
                with fx_traceback.preserve_node_meta(), warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Anomaly Detection has been enabled."
                    )
                    with torch.autograd.detect_anomaly(check_nan=False):
                        out = Interpreter(mod).run(*args[params_len:], **kwargs)
            else:
                out = mod(*args[params_len:], **kwargs)

        if not isinstance(out, (tuple, list)):
            raise RuntimeError(
                "Graph output must be a tuple(). This is so that we can avoid "
                "pytree processing of the outputs. Please change the module to "
                "have tuple outputs or use aot_module instead."
            )
        return out
    return functional_call

# Creates a function that returns flattened inputs and outputs
# Also returns the output tree spec, which is needed to recover the "unflattened"
# output tree structure later.
def create_tree_flattened_fn(fn, args, kwargs=None) -> Tuple[Callable, PytreeThunk]:
    if kwargs is None:
        kwargs = {}
    # Save the args_spec for flat_tensor_args to unflatten while tracing
    _, tensor_args_spec = pytree.tree_flatten((args, kwargs))
    out_spec = PytreeThunk()

    def flat_fn(*flat_args):
        # The input are flattened tensor args. Prepare the args in the
        # order that original function expects. Add static args as well.
        # They will appear as tensor constants in the traced graph.
        nonlocal out_spec
        args, kwargs = pytree.tree_unflatten(flat_args, tensor_args_spec)
        tree_out = fn(*args, **kwargs)
        flat_out, spec = pytree.tree_flatten(tree_out)
        for i in flat_out:
            is_known_type = False
            for j in KNOWN_TYPES:
                if isinstance(i, j):
                    is_known_type = True
                    break
            if not is_known_type:
                raise RuntimeError(
                    f"Found {type(i)} in output, which is not a known type. "
                    "If this type holds tensors, you need to register a pytree for it. "
                    "See https://github.com/pytorch/functorch/issues/475 for a brief "
                    "explanation why. If you don't need to register a pytree, please "
                    "leave a comment explaining your use case and we'll make this more "
                    "ergonomic to deal with"
                )
        out_spec.set(spec)
        return flat_out
    return flat_fn, out_spec

def _graph_input_names(gm):
    return [node.name for node in gm.graph.nodes if node.op == "placeholder"]


def _graph_output_names(gm):
    output_node = next(iter(reversed(gm.graph.nodes)))
    assert output_node.op == "output" and len(output_node.args) == 1
    return_args = output_node.args[0]
    return [getattr(return_arg, "name", None) for return_arg in return_args]


def create_graph_signature(
    fx_g: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec,
    *,
    user_args_flat: List[torch.Tensor],
    params_and_buffers_flat: List[torch.Tensor],
    param_names: List[str],
    buffer_names: List[str],
    trace_joint: bool,
    num_user_fw_outs: Optional[int],
    loss_index: Optional[int],
) -> GraphSignature:

    # Retrieve graph input names
    graph_input_names = _graph_input_names(fx_g)
    # Retrieve graph output names
    graph_output_names = _graph_output_names(fx_g)

    num_params_buffers = len(param_names) + len(buffer_names)
    # We have enough restrictions on the graph (no de-duping, synthetic bases, etc),
    # Such that # graph inps = # user inps + # params + # buffers
    num_user_args = len(graph_input_names) - num_params_buffers

    if trace_joint:
        assert num_user_fw_outs is not None
        num_fw_outs = num_user_fw_outs + fw_metadata.num_mutated_inputs
        backward_output_names = graph_output_names[num_fw_outs:]

        grad_index = itertools.count(0)
        gradients_to_parameters = {
            backward_output_names[next(grad_index)]: param_names[i]
            for i, param in enumerate(params_and_buffers_flat)
            if param.requires_grad
        }

        gradients_to_user_inputs = {
            backward_output_names[next(grad_index)]: graph_input_names[i + len(params_and_buffers_flat)]
            for i, user_input in enumerate(user_args_flat)
            if user_input.requires_grad
        }

        assert len(gradients_to_parameters) + len(gradients_to_user_inputs) == len(
            backward_output_names
        )

        # Check that we have fully accounted for all graph outputs
        backward_signature = BackwardSignature(
            gradients_to_parameters,
            gradients_to_user_inputs,
            graph_output_names[loss_index],
        )
    else:
        backward_signature = None
        num_user_fw_outs = len(graph_output_names) - fw_metadata.num_mutated_inputs

    return GraphSignature.from_tracing_metadata(
        in_spec=in_spec,
        out_spec=out_spec,
        graph_input_names=graph_input_names,
        graph_output_names=graph_output_names,
        view_mutation_metadata=fw_metadata,
        named_parameters=param_names,
        named_buffers=buffer_names,
        num_user_inputs=num_user_args,
        num_user_outputs=num_user_fw_outs,
        loss_index=loss_index,
        backward_signature=backward_signature,
    )

def aot_function(
    fn: Callable,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[Dict] = None,
    num_params_buffers: int = 0,
    keep_inference_input_mutations: bool = False,
    inference_compiler: Optional[Callable] = None,
    *,
    # Whether or not to trace with dynamic shapes
    dynamic=False,
    enable_log=True,
) -> Callable:
    """
    Traces the forward and backward graph of :attr:`fn` using torch dispatch
    mechanism, and then compiles the generated forward and backward graphs
    through :attr:`fw_compiler` and :attr:`bw_compiler`.

    :func:`aot_function` traces the forward and backward graph ahead of time,
    and generates a joint forward and backward graph.  :attr:`partition_fn` is
    then used to separate out forward and backward graphs. The partitioner
    function can be used to perform optimizations such as recomputation. One can
    set `decompositions` dictionary to decompose the operators into a sequence
    of core or simpler operators supported by the backend compilers.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Callable): A Python function that takes one ore more arguments. Must
            return one or more Tensors.
        fw_compiler (Callable): A Python function that accepts an Fx graph with
            Aten ops and input args, and returns a Callable that semantically is
            equivalent to the input Fx graph.
        bw_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph.  Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
        partition_fn (Callable): A Python function that takes a joint forward
            and backward graph, and partitions it into separate forward and
            backward graphs.
        decompositions (Dict): A dictionary to define the decomposition of
            larger Aten ops into simpler or core Aten ops.
        inference_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph. inference_compiler is invoked
            if no autograd is needed. Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
    Returns:
        Returns a ``Callable`` that retains the eager behavior of the original
        :attr:`fn`, but with forward and backward graph compiled via
        :attr:`fw_compile` and :attr:`bw_compile`.

    A simple example usage of :func:`aot_function` is as follows. This example
    will print the forward and backward graphs of the function ``fn``

        >>> fn = lambda x : x.sin().cos()
        >>> def print_compile_fn(fx_module, args):
        >>>     print(fx_module)
        >>>     return fx_module
        >>> aot_fn = aot_function(fn, print_compile_fn)
        >>> x = torch.randn(4, 5, requires_grad=True)
        >>> aot_fn(x)
    """

    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler
    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
        dynamic_shapes=dynamic,
        aot_autograd_arg_pos_to_source=None,
        is_export=False,
        no_tangents=False,
        enable_log=enable_log,
    )
    cached_res = None

    @wraps(fn)
    def returned_function(*args, **kwargs):
        nonlocal cached_res
        # Now flatten the tensor args
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)

        # Compile the function and save it in the cache
        if cached_res is None:
            flat_fn, out_spec = create_tree_flattened_fn(fn, args, kwargs)

            compiled_fn = create_aot_dispatcher_function(
                flat_fn,
                flat_args,
                aot_config,
            )
            cached_res = (compiled_fn, out_spec)

        cached_fn, out_spec = cached_res
        out = cached_fn(flat_args)
        return out_spec.unflatten(out)

    return returned_function


def aot_module(mod: nn.Module, *args, **kwargs) -> nn.Module:
    """
    Traces the forward and backward graph of :attr:`mod` using torch dispatch
    tracing mechanism. It is wrapper function, that underneath uses
    :func:`aot_function` to perform tracing and compilation.

    :func:`aot_module` lifts the parameters and buffers of ``nn.Module`` as inputs
    to a new callable which is then compiled through :func:`aot_function`.

    .. warning::
        This API is experimental and likely to change.

    Args:
        mod (Callable): A ``nn.Module`` module.
        args : args to be passed to :func:`aot_function`
        kwargs : kwargs to be passed to :func:`aot_function`

    Returns:
        Returns a ``nn.Module`` that retains the eager behavior of the original
        :attr:`mod`, but with forward and backward graph compiled.

    """
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)

    def functional_call(named_params, named_buffers, *args, **kwargs):
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)
    compiled_f = aot_function(
        functional_call, *args, num_params_buffers=num_params_buffers, **kwargs
    )

    class AOTModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.orig_module = mod

        def forward(self, *args, **kwargs):
            return compiled_f(
                named_params,
                named_buffers,
                *args,
                **kwargs,
            )

    return AOTModule()


def aot_module_simplified(
    mod: nn.Module,
    args,
    fw_compiler: Callable,
    bw_compiler: Optional[Callable] = None,
    partition_fn: Callable = default_partition,
    decompositions: Optional[Dict] = None,
    keep_inference_input_mutations=False,
    inference_compiler: Optional[Callable] = None,
) -> nn.Module:
    """
    This is the simplified or low overhead version of aot_module. For frontends
    like TorchDynamo, the input functions/modules to AOT are static and have
    unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified` removes these overheads.
    """
    params = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)
    params_len = len(params_flat)

    functional_call = create_functional_call(mod, params_spec, params_len)

    if bw_compiler is None:
        bw_compiler = fw_compiler
    if inference_compiler is None:
        inference_compiler = fw_compiler

    seen_sources = set()

    full_args = []
    # First, the params
    full_args.extend(params_flat)

    if torch._guards.TracingContext.get():
        torch._guards.TracingContext.get().params_flat = params_flat

    aot_autograd_arg_pos_to_source = None
    # Then, the params 1:1 mapped sources, if relevant.
    if hasattr(mod, "_param_name_to_source"):
        aot_autograd_arg_pos_to_source = []
        # We now know this came from dynamo, and (1) we care about guards,
        # so setting up aot_autograd_arg_pos_to_source for downstream dedup guards
        # can now be done safely. (2) Dynamo logic protects the 1:1 sizing below.
        for name in params.keys():
            assert name in mod._param_name_to_source, f"{name} not found."
            source = mod._param_name_to_source[name]
            assert source not in seen_sources, source
            seen_sources.add(source)
            aot_autograd_arg_pos_to_source.append(source)

    # Next, the input args
    full_args.extend(args)

    if hasattr(mod, "graph"):
        # Non dynamo entrypoints can get to here...
        for i, node in enumerate(mod.graph.nodes):
            if node.op == "placeholder":
                if hasattr(node, "_dynamo_source"):
                    # ... but not here!
                    if aot_autograd_arg_pos_to_source is None:
                        aot_autograd_arg_pos_to_source = []
                    source = node._dynamo_source
                    assert source not in seen_sources, source
                    seen_sources.add(source)
                    aot_autograd_arg_pos_to_source.append(source)

    if aot_autograd_arg_pos_to_source is not None:
        assert len(full_args) == len(aot_autograd_arg_pos_to_source)

    dynamic_shapes = False
    for x in full_args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break

    aot_config = AOTConfig(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        inference_compiler=inference_compiler,
        partition_fn=partition_fn,
        decompositions=decompositions,
        num_params_buffers=params_len,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=aot_autograd_arg_pos_to_source,
        is_export=False,
        no_tangents=False,
    )

    with compiled_autograd.disable():
        compiled_fn = create_aot_dispatcher_function(
            functional_call,
            full_args,
            aot_config,
        )

    # TODO: There is something deeply wrong here; compiled_fn running with
    # the boxed calling convention, but aot_module_simplified somehow
    # historically returned a function that was not the boxed calling
    # convention.  This should get fixed...
    def forward(*runtime_args):
        full_args = []
        full_args.extend(params_flat)
        full_args.extend(runtime_args)
        return compiled_fn(full_args)

    # Just for convenience
    forward.zero_grad = mod.zero_grad
    forward.named_parameters = mod.named_parameters
    forward.named_buffers = mod.named_buffers

    return forward

def aot_export_module(
    mod: nn.Module,
    args,
    *,
    decompositions: Optional[Dict] = None,
    # If true, we'll return a joint forward-backward graph,
    # As well as metadata on the loss + gradients in the backward.
    trace_joint: bool,
    # If trace_joint is True, we expect your module to return a scalar loss.
    # Your module can return multiple outputs, so you must specify which output the loss is.
    output_loss_index: Optional[int] = None,
) -> Tuple[torch.fx.GraphModule, GraphSignature]:
    """
    This function takes in a module, and returns:
    (1) an FX graph that can be exported
    (2) some metadata about the graph

    If `trace_joint=True` we will return a joint graph of the forward + backward.

    The traced FX graph will have the following properties compared to the original module:
    (1) Inputs and outputs to the module will be pytree-flattened
    (2) Parameters and buffers on the module will be lifted into graph inputs,
        graph_inputs = (*parameters, *buffers, *user_inputs)
    (3) The graph will be fully functionalized
    (4) Any input mutations will be converted into additional outputs in the graph,
        meaning whoever calls this graph is responsible for applying the mutations
        back to the original inputs.
    (5) If is_joint is provided the graph will return parameter gradients in addition to user outputs.
        The graph output will look like:
        graph_outputs = (*updated_inputs, *user_outputs, *param_gradients)

    There are also several restrictions on what modules can use this API. In particular:
    (1) If trace_joint is specified, we expect the loss function to be **fused**
        into the module forward. One of the outputs to the forward must be a scalar loss,
        which is specified with `output_loss_index`.
        All other outputs to the forward are presumed to not require gradients.
    (2) This API cannot capture optimizers (although in theory we could build an API for this).
    (3) Metadata mutations on params/buffers/inputs are banned.
    (4) Data mutations on anything that requires gradients are banned (parameters)
    (5) If an input is mutated, it is not allowed to alias any other inputs.
    (6) Parameters must not be duplicated.
    """
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat)
    params_len = len(params_and_buffers_flat)

    functional_call = create_functional_call(mod, params_spec, params_len)

    num_fw_outs = None

    if trace_joint:
        # This helper effectively just adds some extra asserts about what the backward will look like:
        # Outputs must include a scalar loss, that we compute gradients w.r.t.
        # We don't compute gradients w.r.t. anything else: so just in case we detach()
        # and other output tensors.
        def fn_to_trace(*args):
            nonlocal num_fw_outs
            out = functional_call(*args)
            if output_loss_index is None:
                raise RuntimeError("""\
If trace_joint=Trueit is required that one of your forward outputs must be a scalar loss.
You must specify the which (index) output is the loss with output_loss_index.""")
            if isinstance(out, (torch.Tensor)):
                out = (out,)
            if not isinstance(out, (tuple, list)):
                raise RuntimeError(f"Expected forward output to be either a tensor or a list/tuple of tensors. found {type(out)}")

            for i, o in enumerate(out):
                # We only want to create a backward graph w.r.t. the loss that the user passed in.
                # This implies that every other output should not require gradients.
                # Instead of making this an error (and forcing the user to detach all other outputs
                # of their forward),
                # we'll automatically detach them here.
                if o.requires_grad and i != output_loss_index:
                    raise RuntimeError(f"""\
Found an output of the forward that requires gradients, that was not the scalar loss.
We require all outputs to the forward that are not the scalar loss to not require gradient,
because we will only compute a backward graph against the scalar loss.
You can fix this by calling .detach() on each of your forward outputs that is not the loss.
You specified that output index {output_loss_index} is the loss, but we found that
the output at index {i} requires gradients.""")
            out_loss = out[output_loss_index]
            num_fw_outs = len(out)
            if not out_loss.requires_grad:
                raise RuntimeError(f"""\
The output at index {output_loss_index} was marked as the loss, but it does not require gradients""")
            if out_loss.numel() != 1:
                raise RuntimeError(f"""\
We require the output marked as the loss (at index {output_loss_index}) to be a scalar, but it has shape {out_loss.shape}""")
            return out
        ctx = nullcontext
    else:
        # Run under no_grad, so our tracing machinery only traces an inference graph.
        ctx = torch.no_grad
        fn_to_trace = functional_call

    full_args = []
    # First, the params
    # NB: It is REQUIRED that parameters come first, Inductor infers "fixed"
    # parameters by looking at the difference in parameter count outside
    # and inside AOTAutograd, and assumes the prefix of arguments are fixed
    # arguments
    full_args.extend(params_and_buffers_flat)
    # Next, the input args
    full_args.extend(args)

    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            fn_to_trace,
            full_args,
            decompositions=decompositions,
            num_params_buffers=len(params_and_buffers_flat),
            no_tangents=True,
        )
    if trace_joint:
        def flattened_joint(*args):
            # The idea here is that the joint graph that AOTAutograd creates has some strict properties:
            # (1) It accepts two arguments (primals, tangents), and pytree_flattens them
            # (2) It returns a tuple of (fw_outs, gradients)
            # This is a very useful convention for anyone who wants to partition the joint graph
            # into a separate forward and backward graph.
            # However,
            # (1) for people exporting a single joint graph, it would be preferable not to have
            #     any pytrees in the graph.
            # (2) We are guaranteed in the aot_export_module case that the forward outputs a loss,
            #     and there are therefore no tangents that are needed to run the joint graph.
            # (3) AOTAutograd creates a grad_input for every input in the forward,
            #     including None's for inputs that are not grad-requiring tensors.
            #     we don't want these in our export graph.
            #     and there are therefore no tangents that are needed to run the joint graph.
            # This function "fixes" both of the above by removing any tangent inputs,
            # and removing pytrees from the original FX graph.
            fake_tangents = [None for _ in range(metadata.num_outputs + metadata.num_mutated_inputs)]
            fw_outs, gradients = fx_g(args, fake_tangents)
            assert len(gradients) == len(args)
            output_gradients = []
            for i, (a, grad) in enumerate(zip(args, gradients)):
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    assert grad is not None, """\
Found a parameter that did not receive a gradient.
"This is most likely a bug, but if this needs to be supported please comment on this Github issue:
https://github.com/pytorch/pytorch/issues/101192
"""
                    output_gradients.append(grad)
                else:
                    assert grad is None
            return *fw_outs, *output_gradients
        fx_g = make_fx(flattened_joint)(*full_args)

    user_args_flat = pytree.arg_tree_leaves(*args)
    return fx_g, create_graph_signature(
        fx_g,
        metadata,
        in_spec,
        out_spec,
        user_args_flat=user_args_flat,
        params_and_buffers_flat=params_and_buffers_flat,
        param_names=list(named_parameters.keys()),
        buffer_names=list(named_buffers.keys()),
        trace_joint=trace_joint,
        num_user_fw_outs=num_fw_outs,
        loss_index=output_loss_index,
    )

def aot_export_joint_simple(
    func: Callable,
    args,
    *,
    trace_joint: bool,
    # It looks like the main consequence of this API is that for dynamic shapes,
    # it will assume that parms/buffers are static.
    # With the new inferred dynamic shapes API, maybe this doesn't matter?
    num_params_buffers: int = 0,
    decompositions: Optional[Dict] = None,
) -> torch.fx.GraphModule:
    """
    A simplified version of export. Used by higher order operators.

    This function makes a high-level "no calling convention changes" guarantee:
    - If no inputs require grad (so we export an inference graph),
      there are *no* calling convention change between the exported graph, and "func".
    - If at least one input requires grad (so we trace out and export a joint fw-bw graph),
      Then if you were partition the graph into a separate forward and backward graph,
      The forward graph will have no calling convention changes compared to "func".

    The above also relies on some strong restrictions around which functions this API accepts:
    (1) `args` cannot contain any pytrees (they must have been pytree_flattened already)
    (2) `func` cannot mutate any inputs
    (3) The outputs of `func` cannot alias any inputs.

    Note: this function is only lightly tested today. It will probably be tested more heavily by higher order ops.
    """
    if trace_joint:
        ctx = nullcontext
    else:
        # Run under no_grad, so our tracing machinery only traces an inference graph.
        ctx = torch.no_grad

    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            func,
            args,
            decompositions=decompositions,
        )
    # At this point, we can just directly return the (joint or inference graph) that we traced.
    # First though: a bunch of assertions to make sure that our graph doesn't require
    # any calling convention changes compared to the original function.
    # These restrictions are *in addition to* the general restrictions on export.

    # No input mutations
    if len([x for x in metadata.input_info if x.mutates_data or x.mutates_metadata]) != 0:
        raise RuntimeError(f"aot_export_joint_simple does not support input mutations. {str(metadata)}")
    # No output aliasing
    if len([x for x in metadata.output_info if x.output_type != OutputType.non_alias]) != 0:
        raise RuntimeError(f"aot_export_joint_simple does not support outputs that alias inputs. {str(metadata)}")
    # No pytrees
    if type(in_spec) == pytree.LeafSpec:
        raise RuntimeError(f"aot_export_joint_simple requires inputs to be a single list/tuple. in_spec={str(in_spec)}")
    if len([x for x in in_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(f"aot_export_joint_simple requires individual inputs not to be pytrees. in_spec={str(in_spec)}")
    if type(out_spec) == pytree.LeafSpec:
        raise RuntimeError(f"aot_export_joint_simple requires outputs to be a single list/tuple. out_spec={str(out_spec)}")
    if len([x for x in out_spec.children_specs if type(x) != pytree.LeafSpec]) != 0:
        raise RuntimeError(f"aot_export_joint_simple requires individual outputs not to be pytrees. out_spec={str(out_spec)}")
    # TODO: we might have to temporarily patch config.functionalize_rng
    # so that it doesn't run when we're exporting a higher order op.

    if config.debug_assert:
        # Smoke test that after partitioning, we can run the forward without any calling convention changes.
        fw_module, bw_module = aot_config.default_partition(
            fx_g, args, num_fwd_outputs=len(fw_metadata.output_infos)
        )
        # Attempt to run the fw_module with the original user inputs
        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            fake_mode = FakeTensorMode()
        with fake_mode:
            fw_module(*args)
    return fx_g

# Private for now because we aren't providing a contract on what to return
# for joint graphs (we could when there's a clearer use case)
# In the future, we may need to add more export API's that provide their own strong guarantees.
# This is meant as a general helper function for handling various export-y use cases.
def _aot_export_function(
    func: Callable,
    args,
    *,
    num_params_buffers: int = 0,
    decompositions: Optional[Dict] = None,
    # If we're exporting a joint graph and we don't want any tangent inputs in the graph
    # (because we are backpropping through a scalar 1 loss),
    # we need to explicitly specify not to include tangents in the graph.
    # It's not enough just to check that our tangent is a scalar, since we also
    # need to know if it is a 1 (no need to make it a graph input), or something else
    # (requiring it to be a graph input).
    # We don't know this info at trace time though, so we need to make it an explicit config.
    no_tangents: bool = False,
) -> Tuple[torch.fx.GraphModule, ViewAndMutationMeta, pytree.TreeSpec, pytree.TreeSpec]:
    dynamic_shapes = False
    for x in args:
        if isinstance(x, FakeTensor):
            dynamic_shapes = x.fake_mode.shape_env is not None
            break

    flat_fn, out_spec = create_tree_flattened_fn(func, args)
    flat_args, in_spec = pytree.tree_flatten(args)

    # The export use case doesn't care about several bits of AOTConfig
    # (1) compilers (we just export the graph)
    # (2) partitioners (export is only full graph, user can partition themselves)
    aot_config = AOTConfig(
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        partition_fn=None,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        # For now there's no use case involving keeping input mutations in the graph
        # (which we can only do in the inference case anyway).
        # We can add this later if we need to.
        keep_inference_input_mutations=False,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=None,
        is_export=True,
        no_tangents=no_tangents,
    )

    fx_g, meta = create_aot_dispatcher_function(
        flat_fn,
        flat_args,
        aot_config,
    )
    return fx_g, meta, in_spec, out_spec.spec


compiled_function = aot_function
compiled_module = aot_module
