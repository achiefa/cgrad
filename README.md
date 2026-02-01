# minigrad++

A minimalistic educational implementation of automatic differentiation (backpropagation) written in pure C.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), this project demonstrates the core concepts of computational graphs, gradient computation, and arena-based memory management in a simple, readable codebase.

## Features

- Tape-based reverse-mode automatic differentiation
- Arena memory allocator with 4KB blocks
- Scalar operations with gradient tracking
- Clean C API with no external dependencies

## Quick Start

```bash
# Build the library and example
make lib
make example

# Run the example
./examples/simple
```

## Usage

```c
#include "cgrad/cgrad.h"

int main(void) {
    // Get the global tape (singleton pattern)
    Tape* tape = tape_get_instance();

    // Create values with gradient tracking enabled
    ValueData* a = value_create(2.0, "a", 1);  // requires_grad = 1
    ValueData* b = value_create(3.0, "b", 1);

    // Build computation graph
    ValueData* c = value_mul(a, b);  // c = a * b = 6.0

    // Backward pass computes gradients
    // dc/da = b = 3.0
    // dc/db = a = 2.0

    printf("c = %f\n", value_get_data(c));     // 6.0
    printf("dc/da = %f\n", value_get_grad(a)); // 3.0
    printf("dc/db = %f\n", value_get_grad(b)); // 2.0

    // Clean up
    tape_destroy_instance();
    return 0;
}
```

## Build Commands

| Command | Description |
|---------|-------------|
| `make help` | Show available targets |
| `make lib` | Build static library (`libcgrad.a`) |
| `make example` | Compile example program |
| `make clean` | Remove build artifacts |
| `make info` | Display platform and compiler info |

## Architecture

### Memory Management (`tape.h` / `tape.c`)

The tape serves two purposes:
1. **Arena allocator** - Pre-allocates memory in 4KB blocks for fast, cache-friendly allocation
2. **Computation graph** - Tracks all nodes for the backward pass

```
Tape
├── blocks[]        # Memory arena (4KB blocks)
├── num_blocks      # Current block count
├── nodes[]         # Pointers to all ValueData nodes
└── num_nodes       # Node count for backward traversal
```

### Value Nodes (`value.h` / `value.c`)

Each `ValueData` represents a node in the computation graph:

```
ValueData
├── data           # Forward pass result (float32)
├── grad           # Accumulated gradient (float32)
├── name[32]       # Optional label for debugging
├── op[8]          # Operation type ("+", "*", etc.)
├── requires_grad  # Whether to compute gradients
├── backward_fn    # Function pointer for backward pass
├── cached_a/b     # Operand values needed during backward
└── children[2]    # Input nodes
```

### Supported Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| `value_add(a, b)` | `a + b` | `da += grad`, `db += grad` |
| `value_mul(a, b)` | `a * b` | `da += b * grad`, `db += a * grad` |

## Project Structure

```
minigrad++/
├── cgrad/
│   ├── cgrad.h     # Main public header
│   ├── tape.h      # Arena allocator interface
│   ├── tape.c      # Arena allocator implementation
│   ├── value.h     # Value operations interface
│   └── value.c     # Value operations implementation
├── examples/
│   └── simple.c    # Basic usage example
├── Makefile
└── README.md
```

## Code Formatting

This project uses `clang-format` for consistent code style.

```bash
./format.sh           # Format all files in-place
./format.sh --check   # Check formatting without modifying
./format.sh --verbose # Format with detailed output
```

### Installing clang-format

```bash
# macOS
brew install clang-format

# Ubuntu/Debian
sudo apt-get install clang-format

# Fedora
sudo dnf install clang-tools-extra
```

## Educational Purpose

This library is designed for learning, not production use. It demonstrates:

- How automatic differentiation works under the hood
- Tape-based gradient accumulation
- Arena memory management patterns in C
- Building computation graphs with function pointers

For production autodiff, consider PyTorch, JAX, or TensorFlow.

## License

GPL-3.0 - See [LICENSE](LICENSE) for details.
