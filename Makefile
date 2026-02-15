# cgrad - pure c automatic differentiation tool
# Makefile

CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm # math library
SRC_FOLDER = cgrad
EX_FOLDER = examples
TEST_FOLDER = tests

# Source files
SRCS = $(SRC_FOLDER)/tape.c $(SRC_FOLDER)/value.c
OBJS = $(SRCS:.c=.o)
EX_SRCS = $(EX_FOLDER)/simple.c
EX_BIN = $(EX_FOLDER)/simple
TEST_SRCS = $(TEST_FOLDER)/main.c
TEST_BIN = $(TEST_FOLDER)/test_runner
LIB = libcgrad.a

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -fsanitize=address

.PHONY: all clean debug lib info help example test

# Default: show available targets
all: help

help:
		@echo "CGrad - Build Targets"
		@echo ""
		@echo "  make          - Show this help message"
		@echo "  make example  - Compile example program(s)"
		@echo "  make clean    - Clean build files"
		@echo "  make info     - Show build configurations"
		@echo "  make lib      - Build static library"
		@echo "  make test     - Build and run unit tests"

example: $(LIB)
		@echo "Compiling example program(s)..."
		$(CC) $(CFLAGS) -I$(SRC_FOLDER) $(EX_SRCS) -L. -lcgrad $(LDFLAGS) -o $(EX_BIN)
		@echo "Built: $(EX_BIN)"

# =============================================================
# Build rules
# =============================================================

lib: $(LIB)

$(LIB): $(OBJS)
		ar rs $@ $^

%.o: %.c
		$(CC) $(CFLAGS) -I$(SRC_FOLDER) -c $< -o $@

# =============================================================
# Tests
# =============================================================
test: $(LIB)
		@echo "Compiling tests..."
		$(CC) $(CFLAGS) -I$(SRC_FOLDER) $(TEST_SRCS) -L. -lcgrad $(LDFLAGS) -o $(TEST_BIN)
		@echo "Running tests..."
		@./$(TEST_BIN)

# =============================================================
# Utilities
# =============================================================
clean:
		rm -rf $(OBJS) $(LIB) $(EX_BIN) $(TEST_BIN)

info:
		@echo "Platform: $(UNAME_S) $(UNAME_M)"
		@echo "Compiler: $(CC)"
		@echo ""
