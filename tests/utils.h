#ifndef CGRAD_TEST_H
#define CGRAD_TEST_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cgrad/cgrad.h"

/* Color definitions */
#define CLR_GREEN "\033[32m"
#define CLR_RED "\033[31m"
#define CLR_YELLOW "\033[33m"
#define CLR_RESET "\033[0m"

/* Per-suite bookkeeping */
static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_current_test_failed = 0;

/* Assertion helpers */
#define ASSERT_NEAR(actual, expected, tol)                                                  \
    do {                                                                                    \
        scalar_t _a = (actual);                                                             \
        scalar_t _e = (expected);                                                           \
        if (fabsf(_a - _e) >= (tol)) {                                                     \
            printf("    " CLR_RED "FAIL" CLR_RESET " %s:%d: ASSERT_NEAR(%s, %s, %s)\n"     \
                   "         got %.8f, expected %.8f (diff %.8e)\n",                        \
                   __FILE__, __LINE__, #actual, #expected, #tol, (double)_a, (double)_e,    \
                   (double)fabsf(_a - _e));                                                 \
            g_current_test_failed = 1;                                                      \
        }                                                                                   \
    } while (0)

#define ASSERT_EQ(actual, expected)                                                         \
    do {                                                                                    \
        long long _a = (long long)(actual);                                                 \
        long long _e = (long long)(expected);                                               \
        if (_a != _e) {                                                                     \
            printf("    " CLR_RED "FAIL" CLR_RESET " %s:%d: ASSERT_EQ(%s, %s)\n"           \
                   "         got %lld, expected %lld\n",                                    \
                   __FILE__, __LINE__, #actual, #expected, _a, _e);                         \
            g_current_test_failed = 1;                                                      \
        }                                                                                   \
    } while (0)

#define ASSERT_TRUE(expr)                                                                   \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            printf("    " CLR_RED "FAIL" CLR_RESET " %s:%d: ASSERT_TRUE(%s)\n",             \
                   __FILE__, __LINE__, #expr);                                              \
            g_current_test_failed = 1;                                                      \
        }                                                                                   \
    } while (0)

#define ASSERT_NOT_NULL(ptr)                                                                \
    do {                                                                                    \
        if ((ptr) == NULL) {                                                                \
            printf("    " CLR_RED "FAIL" CLR_RESET " %s:%d: ASSERT_NOT_NULL(%s)\n",         \
                   __FILE__, __LINE__, #ptr);                                               \
            g_current_test_failed = 1;                                                      \
        }                                                                                   \
    } while (0)

/* test lifecycle */

/* Default tolerance for float comparisons */
#define DEFAULT_TOL 1e-5f

/*
 * RUN_TEST(fn) - runs a single test function.
 *
 * Each test gets a fresh tape. The tape is destroyed after the test
 * regardless of pass/fail so that tests are fully independent.
 */
#define RUN_TEST(fn)                                                \
    do {                                                            \
        g_current_test_failed = 0;                                  \
        tape_get_instance(); /* ensure tape exists */               \
        printf("  %-50s ", #fn);                                   \
        fn();                                                       \
        tape_destroy_instance();                                    \
        g_tests_run++;                                              \
        if (g_current_test_failed) {                                \
            g_tests_failed++;                                       \
            printf(CLR_RED "FAILED" CLR_RESET "\n");                \
        } else {                                                    \
            g_tests_passed++;                                       \
            printf(CLR_GREEN "OK" CLR_RESET "\n");                  \
        }                                                           \
    } while (0)

/*
 * TEST_SUITE(name) - prints a header for a test suite.
 */
#define TEST_SUITE(name) printf("\n" CLR_YELLOW "── %s ──" CLR_RESET "\n", name)

/*
 * TEST_REPORT() - prints a summary and returns an exit code.
 */
#define TEST_REPORT()                                                                   \
    do {                                                                                \
        printf("\n────────────────────────────────────────────\n");                      \
        printf("Tests run: %d  |  " CLR_GREEN "passed: %d" CLR_RESET                   \
               "  |  " CLR_RED "failed: %d" CLR_RESET "\n",                            \
               g_tests_run, g_tests_passed, g_tests_failed);                            \
        printf("────────────────────────────────────────────\n");                        \
    } while (0)

#endif /* CGRAD_TEST_H */
