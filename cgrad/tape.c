/* tape.c - Tape implementation */

#include "tape.h"

#include "value.h"

#include <stdio.h>
#include <stdlib.h>

/* Initial capacities */
#define INITIAL_BLOCKS_CAPACITY 8
#define INITIAL_NODES_CAPACITY  64

/* Global singleton instance */
static Tape *g_tape_instance = NULL;

Tape *tape_create(void) {
    Tape *t = (Tape *)malloc(sizeof(Tape));
    if (!t)
        return NULL;

    /* Initialize with default block capacity */
    t->blocks = (TapeBlock **)malloc(sizeof(TapeBlock *) * INITIAL_BLOCKS_CAPACITY);
    t->num_blocks = 0;
    t->blocks_capacity = INITIAL_BLOCKS_CAPACITY;

    t->nodes = (ValueData **)malloc(sizeof(ValueData *) * INITIAL_NODES_CAPACITY);
    t->num_nodes = 0;
    t->nodes_capacity = INITIAL_NODES_CAPACITY;

    return t;
}

void tape_destroy(Tape *t) {
    if (!t)
        return;

    /* Free all blocks */
    for (size_t i = 0; i < t->num_blocks; i++) {
        free(t->blocks[i]);
    }
    free(t->blocks);
    free(t->nodes);
    free(t);
}

Tape *tape_get_instance(void) {
    if (!g_tape_instance) {
        g_tape_instance = tape_create();
    }
    return g_tape_instance;
}

void tape_destroy_instance(void) {
    if (g_tape_instance) {
        tape_destroy(g_tape_instance);
        g_tape_instance = NULL;
    }
}

void *tape_allocate(Tape *t, size_t size) {
    if (!t)
        return NULL;

    // 8 bytes alignment
    size = (size + 7) & ~7;

    /* Check if a new block is needed */
    if (t->num_blocks == 0 || t->blocks[t->num_blocks - 1]->offset + size > TAPE_BLOCK_SIZE) {

        /* No space in current block, allocate a new one
        In principle == is enough, but just in case */
        if (t->num_blocks >= t->blocks_capacity) {
            size_t new_capacity = t->blocks_capacity * 2;
            TapeBlock **new_blocks =
                (TapeBlock **)realloc(t->blocks, sizeof(TapeBlock *) * new_capacity);
            if (!new_blocks)
                return NULL;
            t->blocks_capacity = new_capacity;
            t->blocks = new_blocks;
        }

        /* Allocate a new block */
        TapeBlock *block = (TapeBlock *)malloc(sizeof(TapeBlock));
        if (!block)
            return NULL;
        block->offset = 0;
        t->blocks[t->num_blocks++] = block;
    }

    /* Move the allocation pointer within the current block */
    TapeBlock *block = t->blocks[t->num_blocks - 1];
    void *ptr = block->data + block->offset;
    block->offset += size;
    return ptr;
}

void tape_register_node(Tape *t, ValueData *node) {
    if (!t || !node)
        return;

    /* Grow nodes array if needed */
    if (t->num_nodes >= t->nodes_capacity) {
        size_t new_capacity = t->nodes_capacity * 2;
        ValueData **new_nodes = (ValueData **)realloc(t->nodes, sizeof(ValueData *) * new_capacity);
        if (!new_nodes)
            return;
        t->nodes_capacity = new_capacity;
        t->nodes = new_nodes;
    }

    t->nodes[t->num_nodes++] = node;
}

size_t tape_num_nodes(const Tape *t) {
    return t ? t->num_nodes : 0;
}

size_t tape_num_blocks(const Tape *t) {
    return t ? t->num_blocks : 0;
}

size_t tape_mem_used(const Tape *t) {
    if (!t)
        return 0;

    size_t total = 0;
    for (size_t i = 0; i < t->num_blocks; i++) {
        total += t->blocks[i]->offset;
    }
    return total;
}

void tape_print_stats(const Tape *t) {
    if (!t) {
        printf("Tape stats: (null tape)\n");
        return;
    }

    printf("Tape stats:\n");
    printf("  Number of nodes: %zu\n", tape_num_nodes(t));
    printf("  Number of blocks: %zu\n", tape_num_blocks(t));
    printf("  Memory used: %zu bytes (%f Mb)\n", tape_mem_used(t),
           tape_mem_used(t) / (1024.0 * 1024.0));
}

/* GraphViz */
void tape_graphviz(Tape *t, const char *filename) {
    /* Buffer to hold the result */
    char dot_filename[256];

    /* Format: filename + ".dot" - max 256 chars (including null terminator) */
    snprintf(dot_filename, sizeof(dot_filename), "%s.dot", filename);

    /* Open file */
    FILE *file = fopen(dot_filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Could not open file %s for writing\n", dot_filename);
        return;
    }

    fprintf(file, "digraph G {\n");
    fprintf(file, "  rankdir=LR;\n"); // Left to right layout
    fprintf(file, "  node [shape=record];\n");

    for (size_t i = 0; i < t->num_nodes; i++) {
        /* For any value in the graph, create a node */
        struct ValueData *v = t->nodes[i];
        fprintf(file, "  node_%p [label=\" %s: %f  grad: %f \"];\n", (void *)v, v->name, v->data,
                v->grad);
        if (v->op[0]) {
            /* If this value is a result of an operation, create an op node */
            fprintf(file, "  node_op_%p [label=\"%s\", shape=circle];\n", (void *)v, v->op);
            /* Connect value node to op node */
            fprintf(file, "  node_op_%p -> node_%p;\n", (void *)v, (void *)v);
        }
    }

    fprintf(file, "}\n");
    fclose(file);

    /* System call to generate a PNG using GraphViz (if installed) */
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "dot -V");
    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "Error: GraphViz command failed\n");
    }

    snprintf(cmd, sizeof(cmd), "graph-easy %s --as=png ", dot_filename);
    system(cmd);
}