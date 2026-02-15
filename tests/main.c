#include "test_binary_ops.h"

int main(void) {
    run_binary_ops_tests();

    TEST_REPORT();
    return g_tests_failed > 0 ? 1 : 0;
}
