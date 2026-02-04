/*
 * simple.c - Basic automatic differentiation example
 *
 * Demonstrates basic arithmetic operations and gradient computation.
*/

#include <stdio.h>
#include "../cgrad/cgrad.h"

int main(void) {
  Tape* tape = tape_get_instance();

  ValueData* a = value_create(2.0, "a", 1);
  ValueData* b = value_create(-3.0, "b", 1);
  ValueData* c = value_create(10.0, "c", 1);
  ValueData* f = value_create(-2.0, "f", 1);

  /* e = a * b */
  ValueData* e = value_mul(a, b);
  value_set_name(e, "e");

  /* d = e + c */
  ValueData* d = value_add(e, c);
  value_set_name(d, "d");

  /* L = d * f */
  ValueData* L = value_mul(d, f);
  value_set_name(L, "L");

  value_backward(L);

  printf("L = %f\n", value_get_data(L));
  printf("Gradients:\n");
  printf("  dL/da = %f (expected: %f)\n", value_get_grad(a), -2.0 * (-3.0));
  printf("  dL/db = %f (expected: %f)\n", value_get_grad(b), -2.0 * 2.0);
  printf("  dL/dc = %f (expected: %f)\n", value_get_grad(c), -2.0);
  printf("  dL/df = %f (expected: %f)\n", value_get_grad(f), 2.0 * (-3.0) + 10.0);

  tape_print_stats(tape);
  tape_graphviz(tape, "simple_graphviz");
  tape_destroy_instance();
  return 0;
}