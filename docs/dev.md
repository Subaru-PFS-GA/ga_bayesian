# Defining a network

Networks are defined by implementing the `model(self, context)` function in a class which should call `context.sample()` as if it was generatively sampling the network. The framework will call this function and pass in a `Model._BuildContext` object which will keep track of the sampled random variables and their dependencies. By tracing the calls to `context.sample()`, the framework can build a directed acyclic graph (DAG) of the network, which can then be used for generating samples and performing Bayesian inference.

Plates can be defined by calling the `Model._BuildContext.plate()` function and wrapping it around the relevant calls using the `with` keyword. This allows the framework to understand that the random variables defined within the plate are N-way copies which are conditionally independent given the variables outside the plate.

Observed variables can be defined by passing `observed=True` to the `Model._BuildContext.sample()` function. This allows the framework to understand that the variable is observed and should be treated as such during inference.

# Tracing random variable dependencies

To trace the calls in `model()`, a tensor wrapped into a `Model._TraceTensor` is returned from every call to `Model._BuildContext.sample()`. This allows the framework to keep track of the dependencies between random variables and build the DAG accordingly.

## Plate crossing

A plate means we need to create N copies of the random variables defined within it. Whenever a random variable is sampled based on another random variable we have to check wether the incoming edges cross a plate boundary. If they do, we need to create N copies of the incoming edge as well, by expanding the incoming tensor to have an extra dimension of size N and copying the values along that dimension. No expanding of the incoming edge is needed if the other node of the incoming is edge is already inside the plate.

# Identifying Markov blankets

# Computing the full conditionals

# Defining Gibbs blocks and Metropolis-Hastings proposals