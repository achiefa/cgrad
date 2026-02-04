#include "value.h"

#include "tape.h"

#include <string.h>

/* Helper function to create a ValueData in the tape */
static ValueData *value_create_internal(Tape *t, scalar_t data, const char *name, int requires_grad,
                                        const char *op, ValueData *child1, ValueData *child2) {

    /* Allocate the node in the memory arena and return the pointer*/
    ValueData *v = (ValueData *)tape_allocate(t, sizeof(ValueData));
    if (!v)
        return NULL;

    /* Initialize ValueData */
    v->data = data;
    v->grad = 0.0;
    v->requires_grad = requires_grad, v->backward_fn = NULL;
    v->cached_a = 0.0;
    v->cached_b = 0.0;
    v->num_children = 0;
    v->children[0] = NULL;
    v->children[1] = NULL;

    if (name && name[0]) {
        strncpy(v->name, name, sizeof(v->name) - 1);
        v->name[sizeof(v->name) - 1] = '\0'; // ensure null termination
    } else {
        v->name[0] = '\0';
    }

    /* Copy op */
    if (op && op[0]) {
        strncpy(v->op, op, sizeof(v->op) - 1);
        v->op[sizeof(v->op) - 1] = '\0';
    } else {
        v->op[0] = '\0';
    }

    /* Set children */
    if (child1) {
        v->children[v->num_children++] = child1;
    }
    if (child2) {
        v->children[v->num_children++] = child2;
    }

    tape_register_node(t, v);
    return v;
}

ValueData *value_create(scalar_t data, const char *name, int requires_grad) {
    Tape *t = tape_get_instance();
    return value_create_internal(t, data, name, requires_grad, "", NULL, NULL);
}

ValueData *value_create_with_tape(struct Tape *t, scalar_t data, const char *name,
                                  int requires_grad) {
    return value_create_internal(t, data, name, requires_grad, "", NULL, NULL);
}

/* Accessors */
scalar_t value_get_data(const ValueData *v) {
    return v ? v->data : 0.0;
}

scalar_t value_get_grad(const ValueData *v) {
    return v ? v->grad : 0.0;
}

const char *value_get_name(const ValueData *v) {
    return v ? v->name : "";
}

int value_requires_grad(const ValueData *v) {
    return v ? v->requires_grad : 0;
}

/* Setters */
void value_set_data(ValueData *v, scalar_t data) {
    if (v)
        v->data = data;
}

void value_set_grad(ValueData *v, scalar_t grad) {
    if (v)
        v->grad = grad;
}

void value_set_name(ValueData *v, const char *name) {
    if (v && name) {
        strncpy(v->name, name, sizeof(v->name) - 1);
        v->name[sizeof(v->name) - 1] = '\0';
    }
}

/* Backward operations for binary operations */

static void backward_add(ValueData *out) {
    /* d/da (a + b) = 1, d/db (a + b) = 1 */
    if (out->children[0])
        out->children[0]->grad += out->grad;
    if (out->children[1])
        out->children[1]->grad += out->grad;
}

static void backward_mul(ValueData *out) {
    /* d/da (a * b) = b, d/db (a * b) = a */
    if (out->children[0])
        out->children[0]->grad += out->cached_a * out->grad;
    if (out->children[1])
        out->children[0]->grad += out->cached_b * out->grad;
}

/* Binary operations */

ValueData *value_add(ValueData *a, ValueData *b) {
    if (!a || !b)
        return NULL;

    Tape *t = tape_get_instance();
    int out_rg = a->requires_grad || b->requires_grad;
    ValueData *out = value_create_internal(t, a->data + b->data, "", out_rg, "+", a, b);

    if (out_rg && out) {
        out->backward_fn = backward_add;
    }
    return out;
}

ValueData *value_mul(ValueData *a, ValueData *b) {
    if (!a || !b)
        return NULL;

    Tape *t = tape_get_instance();
    int out_rg = a->requires_grad || b->requires_grad;
    ValueData *out = value_create_internal(t, a->data * b->data, "", out_rg, "*", a, b);

    if (out_rg && out) {
        out->backward_fn = backward_mul;
        /* Cache values needed for backward pass */
        out->cached_a = b->data;
        out->cached_b = a->data;
    }

    return out;
}

/* Backward pass */
void value_backward(ValueData *v) {
    if (!v) return;

    /* Set gradient of output to 1.0 */
    v->grad = 1.0;

    /* Run backward pass on tape */
    Tape* t = tape_get_instance();
    tape_backward(t);
}