# quantum-computing-simulator


## Goals

- Implement a quantum computing simulator in order to help us understand quantum mechanics / quantum phenomena better.
- Simulate Shor's algorithm for factoring numbers.
- Visualize quantum circuits to help explain what's going on.

## Todo

- `QuantumCircuit`
    - Reimplement the interference example with `QuantumCircuit`
    - Implement Quantum Fourier Transform and its inverse


## Concepts

### Qubits

Example Qubit State:

phi = 1/sqrt(2) * (|0> + |1>)

If measured, Born Rule means that P(0) = 1/2, P(1) = 1/2.  

In general:
phi = alpha * |0> + beta * |1>
where alpha^2 + beta^2 = 1

So, we can put alpha, beta, in a vector and call that the state of our Qubit.

(alpha, beta)^T on the unit sphere is an arbitrary state.


For multiple qubits, we have multiple vectors, but it's convenient to put them in one big vector and apply it.

e.g. CNOT

|00> -> |00>
|01> -> |01>
|10> -> |11>
|11> -> |10>

We can represent this as a 4x4 matrix operating on a 4-component vector.  

e.g.

|10> = [0, 1, 1, 0]^T

