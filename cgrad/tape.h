/*
Memory pool for memory allocations of the computational graph.
*/

#ifndef CGRAD_TAPE_H
#define CGRAD_TAPE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
struct ValueData;

/* Memory block for arena allocation */
#define TAPE_BLOCK_SIZE 4096 // 4KB blocks

typedef struct TapeBlock {
    uint8_t data[TAPE_BLOCK_SIZE];
    size_t offset;
} TapeBlock;

typedef struct Tape {
    TapeBlock **blocks;     // Array of block pointers
    size_t num_blocks;      // Current count
    size_t blocks_capacity; // Allocated capacity

    struct ValueData **nodes; // Array of node pointers
    size_t num_nodes;
    size_t nodes_capacity;
} Tape;

/* Tape lifecycle management */
Tape *tape_create(void);
void tape_destroy(Tape *t);

/* Singleton accessor */
Tape *tape_get_instance(void);
void tape_destroy_instance(void);

/* Memory allocation */
void *tape_allocate(Tape *t, size_t size);

/* Node management */
void tape_register_node(Tape *t, struct ValueData *node);

/* Backward pass */
void tape_backward(Tape *t);
void tape_zero_grad(Tape *t);

/* Statistics */
size_t tape_num_nodes(const Tape *t);
size_t tape_num_blocks(const Tape *t);
size_t tape_mem_used(const Tape *t);
void tape_print_stats(const Tape *t);

/* GraphViz */
void tape_graphviz(Tape *t, const char *filename);

#ifdef __cplusplus
}
#endif

#endif // CGRAD_TAPE_H