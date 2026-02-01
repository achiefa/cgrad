#ifndef CGRAD_H
#define CGRAD_H

/* 
 * cgrad: A pure C implementation of automatic differentiation
 *
 * This library is a simple and educative implementation
 * of the tape-based backward automatic differentiation. Other
 * than simple arithmetic operations, it also implements neural
 * network components (MLP, layers, neurons)
 * 
 * Usage:
 *    #include "minigrad.h"
 *    // Create values
 *    ValueData* a = value_create(2.0, "a", 1);  // requires_grad = 1
 *    ValueData* b = value_create(3.0, "b", 1);
 *
 *    // Compute
 *    ValueData* c = value_mul(a, b);
 *
 *    // Backward pass
 *    value_backward(c);
 *
 *    // Get gradients
 *    printf("da = %f\n", value_get_grad(a));  // 3.0
 *    printf("db = %f\n", value_get_grad(b));  // 2.0
 *
 *    // Clean up
 *    Tape* tape = tape_get_instance();
 *    tape_clear(tape);
 */

#include "tape.h"
#include "value.h"


#endif // CGRAD_H