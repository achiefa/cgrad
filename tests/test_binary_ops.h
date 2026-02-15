#ifndef CGRAD_TEST_BINARYOP
#define CGRAD_TEST_BINARYOP

#include "utils.h"

/* ================================================================
 *  Forward-pass tests  (value correctness only)
 * ================================================================ */

void test_add_forward(void) {
    ValueData *a = value_create(2.0f, "a", 0);
    ValueData *b = value_create(3.0f, "b", 0);
    ValueData *c = value_add(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_NEAR(value_get_data(c), 5.0f, DEFAULT_TOL);
}

void test_add_forward_negative(void) {
    ValueData *a = value_create(-4.0f, "a", 0);
    ValueData *b = value_create(7.0f, "b", 0);
    ValueData *c = value_add(a, b);
    ASSERT_NEAR(value_get_data(c), 3.0f, DEFAULT_TOL);
}

void test_sub_forward(void) {
    ValueData *a = value_create(10.0f, "a", 0);
    ValueData *b = value_create(4.0f, "b", 0);
    ValueData *c = value_sub(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_NEAR(value_get_data(c), 6.0f, DEFAULT_TOL);
}

void test_sub_forward_negative_result(void) {
    ValueData *a = value_create(2.0f, "a", 0);
    ValueData *b = value_create(5.0f, "b", 0);
    ValueData *c = value_sub(a, b);
    ASSERT_NEAR(value_get_data(c), -3.0f, DEFAULT_TOL);
}

void test_mul_forward(void) {
    ValueData *a = value_create(3.0f, "a", 0);
    ValueData *b = value_create(4.0f, "b", 0);
    ValueData *c = value_mul(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_NEAR(value_get_data(c), 12.0f, DEFAULT_TOL);
}

void test_mul_forward_by_zero(void) {
    ValueData *a = value_create(5.0f, "a", 0);
    ValueData *b = value_create(0.0f, "b", 0);
    ValueData *c = value_mul(a, b);
    ASSERT_NEAR(value_get_data(c), 0.0f, DEFAULT_TOL);
}

void test_div_forward(void) {
    ValueData *a = value_create(10.0f, "a", 0);
    ValueData *b = value_create(4.0f, "b", 0);
    ValueData *c = value_div(a, b);
    ASSERT_NOT_NULL(c);
    ASSERT_NEAR(value_get_data(c), 2.5f, DEFAULT_TOL);
}

void test_div_forward_negative(void) {
    ValueData *a = value_create(6.0f, "a", 0);
    ValueData *b = value_create(-3.0f, "b", 0);
    ValueData *c = value_div(a, b);
    ASSERT_NEAR(value_get_data(c), -2.0f, DEFAULT_TOL);
}

/* ================================================================
 *  Gradient tests  (single op, backward)
 * ================================================================ */

void test_add_backward(void) {
    /* L = a + b  =>  dL/da = 1, dL/db = 1 */
    ValueData *a = value_create(2.0f, "a", 1);
    ValueData *b = value_create(-3.0f, "b", 1);
    ValueData *L = value_add(a, b);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), -1.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 1.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), 1.0f, DEFAULT_TOL);
}

void test_sub_backward(void) {
    /* L = a - b  =>  dL/da = 1, dL/db = -1 */
    ValueData *a = value_create(5.0f, "a", 1);
    ValueData *b = value_create(3.0f, "b", 1);
    ValueData *L = value_sub(a, b);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 2.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 1.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), -1.0f, DEFAULT_TOL);
}

void test_mul_backward(void) {
    /* L = a * b  =>  dL/da = b, dL/db = a */
    ValueData *a = value_create(2.0f, "a", 1);
    ValueData *b = value_create(-3.0f, "b", 1);
    ValueData *L = value_mul(a, b);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), -6.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), -3.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), 2.0f, DEFAULT_TOL);
}

void test_div_backward(void) {
    /* L = a / b  =>  dL/da = 1/b, dL/db = -a/b^2 */
    ValueData *a = value_create(6.0f, "a", 1);
    ValueData *b = value_create(3.0f, "b", 1);
    ValueData *L = value_div(a, b);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 2.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 1.0f / 3.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), -6.0f / 9.0f, DEFAULT_TOL);
}

/* ================================================================
 *  Chained / composite expression tests
 * ================================================================ */

void test_chain_add_mul(void) {
    /* L = (a + b) * c
     * dL/da = c,  dL/db = c,  dL/dc = a + b */
    ValueData *a = value_create(2.0f, "a", 1);
    ValueData *b = value_create(3.0f, "b", 1);
    ValueData *c = value_create(4.0f, "c", 1);
    ValueData *sum = value_add(a, b);
    ValueData *L = value_mul(sum, c);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 20.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 4.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), 4.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(c), 5.0f, DEFAULT_TOL);
}

void test_chain_mul_add(void) {
    /* L = (a * b) + c
     * dL/da = b,  dL/db = a,  dL/dc = 1 */
    ValueData *a = value_create(2.0f, "a", 1);
    ValueData *b = value_create(-3.0f, "b", 1);
    ValueData *c = value_create(10.0f, "c", 1);
    ValueData *prod = value_mul(a, b);
    ValueData *L = value_add(prod, c);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 4.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), -3.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), 2.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(c), 1.0f, DEFAULT_TOL);
}

void test_chain_complex(void) {
    /* L = ((a * b) + c) * f
     * a=2, b=-3, c=10, f=-2  =>  L = ((-6)+10)*(-2) = -8
     * dL/da = b*f = 6,  dL/db = a*f = -4,  dL/dc = f = -2,  dL/df = a*b+c = 4 */
    ValueData *a = value_create(2.0f, "a", 1);
    ValueData *b = value_create(-3.0f, "b", 1);
    ValueData *c = value_create(10.0f, "c", 1);
    ValueData *f = value_create(-2.0f, "f", 1);

    ValueData *e = value_mul(a, b);
    ValueData *d = value_add(e, c);
    ValueData *L = value_mul(d, f);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), -8.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 6.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), -4.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(c), -2.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(f), 4.0f, DEFAULT_TOL);
}

void test_chain_div_sub(void) {
    /* L = (a * b) / c - f
     * a=2, b=-3, c=10, f=-2
     * L = -6/10 - (-2) = -0.6 + 2 = 1.4
     * dL/da = b/c = -0.3,  dL/db = a/c = 0.2
     * dL/dc = -(a*b)/c^2 = 6/100 = 0.06,  dL/df = -1 */
    ValueData *a = value_create(2.0f, "a", 1);
    ValueData *b = value_create(-3.0f, "b", 1);
    ValueData *c = value_create(10.0f, "c", 1);
    ValueData *f = value_create(-2.0f, "f", 1);

    ValueData *e = value_mul(a, b);
    ValueData *d = value_div(e, c);
    ValueData *L = value_sub(d, f);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 1.4f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), -0.3f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(b), 0.2f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(c), 0.06f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(f), -1.0f, DEFAULT_TOL);
}

/* ================================================================
 *  Scalar-on-left operation tests
 * ================================================================ */

void test_scalar_add_value(void) {
    /* L = 5 + a  =>  dL/da = 1 */
    ValueData *a = value_create(3.0f, "a", 1);
    ValueData *L = scalar_add_value(5.0f, a);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 8.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 1.0f, DEFAULT_TOL);
}

void test_scalar_sub_value(void) {
    /* L = 5 - a  =>  dL/da = -1 */
    ValueData *a = value_create(3.0f, "a", 1);
    ValueData *L = scalar_sub_value(5.0f, a);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 2.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), -1.0f, DEFAULT_TOL);
}

void test_scalar_div_value(void) {
    /* L = 6 / a  =>  dL/da = -6/a^2 = -6/9 */
    ValueData *a = value_create(3.0f, "a", 1);
    ValueData *L = scalar_div_value(6.0f, a);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 2.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), -6.0f / 9.0f, DEFAULT_TOL);
}

/* ================================================================
 *  Edge cases
 * ================================================================ */

void test_add_null_args(void) {
    ValueData *a = value_create(1.0f, "a", 0);
    ASSERT_TRUE(value_add(NULL, a) == NULL);
    ASSERT_TRUE(value_add(a, NULL) == NULL);
    ASSERT_TRUE(value_add(NULL, NULL) == NULL);
}

void test_mul_null_args(void) {
    ValueData *a = value_create(1.0f, "a", 0);
    ASSERT_TRUE(value_mul(NULL, a) == NULL);
    ASSERT_TRUE(value_mul(a, NULL) == NULL);
}

void test_no_grad_propagation(void) {
    /* When requires_grad=0 for both inputs, backward_fn should be NULL */
    ValueData *a = value_create(2.0f, "a", 0);
    ValueData *b = value_create(3.0f, "b", 0);
    ValueData *c = value_add(a, b);
    ASSERT_NEAR(value_get_data(c), 5.0f, DEFAULT_TOL);
    ASSERT_TRUE(c->backward_fn == NULL);
}

void test_same_value_add(void) {
    /* L = a + a  =>  dL/da = 2 */
    ValueData *a = value_create(3.0f, "a", 1);
    ValueData *L = value_add(a, a);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 6.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 2.0f, DEFAULT_TOL);
}

void test_same_value_mul(void) {
    /* L = a * a  =>  dL/da = 2a = 6 */
    ValueData *a = value_create(3.0f, "a", 1);
    ValueData *L = value_mul(a, a);
    value_backward(L);

    ASSERT_NEAR(value_get_data(L), 9.0f, DEFAULT_TOL);
    ASSERT_NEAR(value_get_grad(a), 6.0f, DEFAULT_TOL);
}

/* ================================================================
 *  Suite runner
 * ================================================================ */

void run_binary_ops_tests(void) {
    TEST_SUITE("Binary Ops - Forward Pass");
    RUN_TEST(test_add_forward);
    RUN_TEST(test_add_forward_negative);
    RUN_TEST(test_sub_forward);
    RUN_TEST(test_sub_forward_negative_result);
    RUN_TEST(test_mul_forward);
    RUN_TEST(test_mul_forward_by_zero);
    RUN_TEST(test_div_forward);
    RUN_TEST(test_div_forward_negative);

    TEST_SUITE("Binary Ops - Backward Pass (Gradients)");
    RUN_TEST(test_add_backward);
    RUN_TEST(test_sub_backward);
    RUN_TEST(test_mul_backward);
    RUN_TEST(test_div_backward);

    TEST_SUITE("Binary Ops - Chained Expressions");
    RUN_TEST(test_chain_add_mul);
    RUN_TEST(test_chain_mul_add);
    RUN_TEST(test_chain_complex);
    RUN_TEST(test_chain_div_sub);

    TEST_SUITE("Scalar-on-Left Operations");
    RUN_TEST(test_scalar_add_value);
    RUN_TEST(test_scalar_sub_value);
    RUN_TEST(test_scalar_div_value);

    TEST_SUITE("Edge Cases");
    RUN_TEST(test_add_null_args);
    RUN_TEST(test_mul_null_args);
    RUN_TEST(test_no_grad_propagation);
    RUN_TEST(test_same_value_add);
    RUN_TEST(test_same_value_mul);
}

#endif /* CGRAD_TEST_BINARYOP */
