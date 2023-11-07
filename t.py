import torch
from torch._dynamo import compiled_autograd
from torch import _inductor as inductor
from torch.autograd import Function
from torch._dynamo.utils import counters

def compiler_fn(gm):
    """Same as torch.compile() but counts number of compiles"""

    def inner_compiler(gm_, example_inputs_):
        counters["compiled_autograd"]["compiles"] += 1
        return inductor.compile(gm_, example_inputs_)

    return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)

def check_output_and_recompiles(fn, count=1):
    with torch.autograd.set_multithreading_enabled(False):
        torch._dynamo.reset()
        counters["compiled_autograd"].clear()
        torch.manual_seed(123)
        expected = list(fn())
        torch.manual_seed(123)
        with compiled_autograd.enable(compiler_fn):
            actual = list(fn())

        assert len(expected) == len(actual)
        for i in range(len(expected)):
            assert torch.equal(expected[i], actual[i])
        assert counters["compiled_autograd"]["captures"] == count
        assert counters["compiled_autograd"]["compiles"] == count

# test_accumulate_without_zero
def fn():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        # torch.nn.ReLU(),
    )
    opt_model = torch.compile(model, dynamic=True)

    # for _ in range(1):
    x = torch.randn([10, 4])
    result = opt_model(x).sum()
    result.backward()
    yield model[0].weight.grad.clone()
    yield model[0].bias.grad.clone()

def custom_fn():
    class LinearFunction(Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight, bias)
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_variables

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output.sum(0)

            return grad_input, grad_weight, grad_bias

    model = LinearFunction.apply

    linear = torch.nn.Linear(4, 4)
    x = torch.randn([10, 4])

    def call_model():
        result = model(x, linear.weight, linear.bias).sum()
        result.backward()

    opt_call_model = torch.compile(call_model, dynamic=True)
    opt_call_model()

    yield linear.weight.grad.clone()
    yield linear.bias.grad.clone()


if __name__ == "__main__":
    # print("not using custom function")
    # check_output_and_recompiles(fn)
    print("using custom function")
    check_output_and_recompiles(custom_fn)
