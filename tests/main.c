/*
 * main.c - Unit tests for CGrad library
 *
 * Tests for automatic differentiation operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../cgrad/cgrad.h"

/* Test counter */
static int tests_passed = 0;
static int tests_failed = 0;

/* Helper macros */
#define ASSERT_FLOAT_EQ(actual, expected, tolerance) \
    do { \
        float diff = fabs((actual) - (expected)); \
        if (diff > (tolerance)) { \
            printf("  FAIL: Expected %f, got %f (diff: %f)\n", \
                   (float)(expected), (float)(actual), diff); \
            tests_failed++; \
            return 0; \
        } \
    } while (0)

#define ASSERT_EQ(actual, expected) \
    do { \
        if ((actual) != (expected)) { \
            printf("  FAIL: Expected %d, got %d\n", (int)(expected), (int)(actual)); \
            tests_failed++; \
            return 0; \
        } \
    } while (0)

#define ASSERT_STR_EQ(actual, expected) \
    do { \
        if (strcmp((actual), (expected)) != 0) { \
            printf("  FAIL: Expected \"%s\", got \"%s\"\n", (expected), (actual)); \
            tests_failed++; \
            return 0; \
        } \
    } while (0)

#define TEST(name) \
    static int test_##name(void); \
    static int test_##name(void)

#define RUN_TEST(name) \
    do { \
        printf("Running " #name "...\n"); \
        if (test_##name()) { \
            printf("  PASS\n"); \
            tests_passed++; \
        } \
    } while (0)

/* Test: Tape creation and destruction */
TEST(tape_create_destroy) {
    Tape* tape = tape_create();
    ASSERT_EQ(tape != NULL, 1);
    ASSERT_EQ(tape_num_nodes(tape), 0);
    tape_destroy(tape);
    return 1;
}

/* Test: Tape singleton */
TEST(tape_singleton) {
    Tape* tape1 = tape_get_instance();
    Tape* tape2 = tape_get_instance();
    ASSERT_EQ(tape1 == tape2, 1);
    tape_destroy_instance();
    return 1;
}

/* Test: Value creation */
TEST(value_create_basic) {
    Tape* tape = tape_get_instance();
    
    ValueData* v = value_create(5.0, "test", 1);
    ASSERT_EQ(v != NULL, 1);
    ASSERT_FLOAT_EQ(value_get_data(v), 5.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(v), 0.0, 1e-6);
    ASSERT_STR_EQ(value_get_name(v), "test");
    ASSERT_EQ(value_requires_grad(v), 1);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Value setters */
TEST(value_setters) {
    Tape* tape = tape_get_instance();
    
    ValueData* v = value_create(1.0, "original", 1);
    value_set_data(v, 2.0);
    value_set_grad(v, 3.0);
    value_set_name(v, "updated");
    
    ASSERT_FLOAT_EQ(value_get_data(v), 2.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(v), 3.0, 1e-6);
    ASSERT_STR_EQ(value_get_name(v), "updated");
    
    tape_destroy_instance();
    return 1;
}

/* Test: Addition operation */
TEST(value_add) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 1);
    ValueData* b = value_create(3.0, "b", 1);
    ValueData* c = value_add(a, b);
    
    ASSERT_FLOAT_EQ(value_get_data(c), 5.0, 1e-6);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 1.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(b), 1.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Subtraction operation */
TEST(value_sub) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(5.0, "a", 1);
    ValueData* b = value_create(3.0, "b", 1);
    ValueData* c = value_sub(a, b);
    
    ASSERT_FLOAT_EQ(value_get_data(c), 2.0, 1e-6);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 1.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(b), -1.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Multiplication operation */
TEST(value_mul) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 1);
    ValueData* b = value_create(3.0, "b", 1);
    ValueData* c = value_mul(a, b);
    
    ASSERT_FLOAT_EQ(value_get_data(c), 6.0, 1e-6);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 3.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(b), 2.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Division operation */
TEST(value_div) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(6.0, "a", 1);
    ValueData* b = value_create(2.0, "b", 1);
    ValueData* c = value_div(a, b);
    
    ASSERT_FLOAT_EQ(value_get_data(c), 3.0, 1e-6);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 0.5, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(b), -1.5, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Scalar addition */
TEST(scalar_add_value) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(3.0, "a", 1);
    ValueData* b = scalar_add_value(2.0, a);
    
    ASSERT_FLOAT_EQ(value_get_data(b), 5.0, 1e-6);
    
    value_backward(b);
    ASSERT_FLOAT_EQ(value_get_grad(a), 1.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Scalar multiplication */
TEST(scalar_mul_value) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(3.0, "a", 1);
    ValueData* b = scalar_mul_value(2.0, a);
    
    ASSERT_FLOAT_EQ(value_get_data(b), 6.0, 1e-6);
    
    value_backward(b);
    ASSERT_FLOAT_EQ(value_get_grad(a), 2.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Complex computation graph */
TEST(complex_graph) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 1);
    ValueData* b = value_create(-3.0, "b", 1);
    ValueData* c = value_create(10.0, "c", 1);
    ValueData* f = value_create(-2.0, "f", 1);
    
    /* e = a * b */
    ValueData* e = value_mul(a, b);
    
    /* d = e + c */
    ValueData* d = value_add(e, c);
    
    /* L = d * f */
    ValueData* L = value_mul(d, f);
    
    ASSERT_FLOAT_EQ(value_get_data(L), -8.0, 1e-6);
    
    value_backward(L);
    
    ASSERT_FLOAT_EQ(value_get_grad(a), 6.0, 1e-6);  /* b * f */
    ASSERT_FLOAT_EQ(value_get_grad(b), -4.0, 1e-6); /* a * f */
    ASSERT_FLOAT_EQ(value_get_grad(c), -2.0, 1e-6); /* f */
    ASSERT_FLOAT_EQ(value_get_grad(f), 4.0, 1e-6);  /* d */
    
    tape_destroy_instance();
    return 1;
}

/* Test: Multiple backward passes */
TEST(multiple_backward) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 1);
    ValueData* b = value_create(3.0, "b", 1);
    ValueData* c = value_mul(a, b);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 3.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(b), 2.0, 1e-6);
    
    /* Clear and do another computation */
    tape_clear(tape);
    
    a = value_create(4.0, "a2", 1);
    b = value_create(5.0, "b2", 1);
    c = value_add(a, b);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 1.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(b), 1.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Tape clear functionality */
TEST(tape_clear) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(1.0, "a", 1);
    ValueData* b = value_create(2.0, "b", 1);
    ValueData* c = value_add(a, b);
    
    size_t nodes_before = tape_num_nodes(tape);
    ASSERT_EQ(nodes_before > 0, 1);
    
    tape_clear(tape);
    
    size_t nodes_after = tape_num_nodes(tape);
    ASSERT_EQ(nodes_after, 0);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Zero gradient functionality */
TEST(tape_zero_grad) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 1);
    ValueData* b = value_create(3.0, "b", 1);
    ValueData* c = value_mul(a, b);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 3.0, 1e-6);
    
    tape_zero_grad(tape);
    ASSERT_FLOAT_EQ(value_get_grad(a), 0.0, 1e-6);
    ASSERT_FLOAT_EQ(value_get_grad(b), 0.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Requires grad flag */
TEST(requires_grad_false) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 0);  /* No grad */
    ValueData* b = value_create(3.0, "b", 1);  /* With grad */
    ValueData* c = value_mul(a, b);
    
    ASSERT_FLOAT_EQ(value_get_data(c), 6.0, 1e-6);
    
    value_backward(c);
    ASSERT_FLOAT_EQ(value_get_grad(a), 0.0, 1e-6);  /* Should be 0 */
    ASSERT_FLOAT_EQ(value_get_grad(b), 2.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Division with complex expression */
TEST(division_complex) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 1);
    ValueData* b = value_create(-3.0, "b", 1);
    ValueData* c = value_create(10.0, "c", 1);
    
    /* e = a * b */
    ValueData* e = value_mul(a, b);
    
    /* d = e / c */
    ValueData* d = value_div(e, c);
    
    ASSERT_FLOAT_EQ(value_get_data(d), -0.6, 1e-6);
    
    value_backward(d);
    
    ASSERT_FLOAT_EQ(value_get_grad(a), -0.3, 1e-6);  /* b / c */
    ASSERT_FLOAT_EQ(value_get_grad(b), 0.2, 1e-6);   /* a / c */
    ASSERT_FLOAT_EQ(value_get_grad(c), 0.06, 1e-6);  /* -(a*b)/(c*c) */
    
    tape_destroy_instance();
    return 1;
}

/* Test: Scalar subtraction */
TEST(scalar_sub_value) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(3.0, "a", 1);
    ValueData* b = scalar_sub_value(5.0, a);  /* 5 - a */
    
    ASSERT_FLOAT_EQ(value_get_data(b), 2.0, 1e-6);
    
    value_backward(b);
    ASSERT_FLOAT_EQ(value_get_grad(a), -1.0, 1e-6);
    
    tape_destroy_instance();
    return 1;
}

/* Test: Scalar division */
TEST(scalar_div_value) {
    Tape* tape = tape_get_instance();
    
    ValueData* a = value_create(2.0, "a", 1);
    ValueData* b = scalar_div_value(8.0, a);  /* 8 / a */
    
    ASSERT_FLOAT_EQ(value_get_data(b), 4.0, 1e-6);
    
    value_backward(b);
    ASSERT_FLOAT_EQ(value_get_grad(a), -2.0, 1e-6);  /* -8/(a^2) */
    
    tape_destroy_instance();
    return 1;
}

/* Main test runner */
int main(void) {
    printf("=================================\n");
    printf("CGrad Unit Tests\n");
    printf("=================================\n\n");
    
    /* Tape tests */
    RUN_TEST(tape_create_destroy);
    RUN_TEST(tape_singleton);
    RUN_TEST(tape_clear);
    RUN_TEST(tape_zero_grad);
    
    /* Value tests */
    RUN_TEST(value_create_basic);
    RUN_TEST(value_setters);
    RUN_TEST(requires_grad_false);
    
    /* Operation tests */
    RUN_TEST(value_add);
    RUN_TEST(value_sub);
    RUN_TEST(value_mul);
    RUN_TEST(value_div);
    
    /* Scalar operation tests */
    RUN_TEST(scalar_add_value);
    RUN_TEST(scalar_mul_value);
    RUN_TEST(scalar_sub_value);
    RUN_TEST(scalar_div_value);
    
    /* Complex tests */
    RUN_TEST(complex_graph);
    RUN_TEST(division_complex);
    RUN_TEST(multiple_backward);
    
    /* Summary */
    printf("\n=================================\n");
    printf("Test Results:\n");
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("=================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
