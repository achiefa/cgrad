#ifndef CGRAD_VALUE_H
#define CGRAD_VALUE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Scalar type - fixed to float32 */
typedef float scalar_t;

/* Backward function pointer type */
typedef void (*BackwardFn)(struct ValueData *output);

/* Computation graph node */
typedef struct ValueData {
    scalar_t data;
    scalar_t grad;
    char name[32];
    char op[8];
    int requires_grad;

    BackwardFn backward_fn;

    // Cache for backward pass
    scalar_t cached_a; // Value needed to compute a's gradient
    scalar_t cached_b; // Value needed to compute b's gradient

    // Children nodes
    struct ValueData *children[2];
    size_t num_children; // TODO check if needed
} ValueData;

/* Value creation */
ValueData *value_create(scalar_t data, const char *name, int required_grad);
ValueData *value_create_with_tape(struct Tape *t, scalar_t data, const char *name,
                                  int required_grad);

/* Value creation */
scalar_t value_get_data(const ValueData *v);
scalar_t value_get_grad(const ValueData *v);
const char *value_get_name(const ValueData *v);
int value_requires_grad(const ValueData *v);

/* Setters */
void value_set_data(ValueData *v, scalar_t data);
void value_set_grad(ValueData *v, scalar_t grad);
void value_set_name(ValueData *v, const char *name);

/* Binary operations */
ValueData *value_add(ValueData *a, ValueData *b);
ValueData *value_mul(ValueData *a, ValueData *b);

#ifdef __cplusplus
}
#endif

#endif // CGRAD_VALUE_H