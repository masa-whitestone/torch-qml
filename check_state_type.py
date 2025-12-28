import torch
import torchqml as tq
import cudaq
import numpy as np

tq.set_backend("qpp-cpu")
q = tq.qvector(2, 1)
tq.h(q[0])
tq.cx(q[0], q[1])

# Build kernel manually to check native cudaq return
kernel, params = tq.utils.kernel_builder.build_circuit_kernel(2, q._operations)
# Pass parameters to kernel (params is a list of Parameter objects, we need values)
# For this simple circuit, params might be empty, but build_circuit_kernel returns them.
# We need to invoke kernel WITH arguments if it expects them.
# Actually, qvector operations might have recorded parameters.
# Let's just create a kernel without parameters for this test.

import cudaq
kernel = cudaq.make_kernel()
qubit = kernel.qalloc(2)
kernel.h(qubit[0])
kernel.cx(qubit[0], qubit[1])

state = cudaq.get_state(kernel)

print(f"State type: {type(state)}")
print(f"Dir state: {dir(state)}")
print(f"Is on GPU: {state.is_on_gpu()}")

try:
    tensor = state.getTensor()
    print(f"getTensor type: {type(tensor)}")
    print(f"getTensor content: {tensor}")
except Exception as e:
    print(f"getTensor failed: {e}")

try:
    # check amplitudes
    amps = state.amplitudes()
    print(f"Amplitudes type: {type(amps)}")
except:
    pass
