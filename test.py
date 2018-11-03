import torch
print(torch.__version__)
def fn(x):
    result = x[0]
    for i in range(x.size(0)):
        result = result * x[i]
    return result

inputs = (torch.rand(3, 4, 5),)
check_inputs = [(torch.rand(4, 5, 6),), (torch.rand(2, 3, 4),)]

scripted_fn = torch.jit.script(fn)
print(scripted_fn.graph)

for input_tuple in [inputs] + check_inputs:
    torch.testing.assert_allclose(fn(*input_tuple), scripted_fn(*input_tuple))