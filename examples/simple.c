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

  tape_print_stats(tape);
  tape_destroy_instance();
  return 0;
}