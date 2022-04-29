CC = gcc -O2
CCFLAGS = -Wall -pedantic
DBGFLAGS = -g
LIBOCL = -lOpenCL
LIBTHREAD = -lpthread
LIBMATH = -lm

TARGETS = libacmatch.a ocl_aho_grep 
UNIT_TESTS = databuf_test compact_array_test

all: $(TARGETS)

unit_tests: $(UNIT_TESTS)

ocl_aho_grep: ocl_aho_grep.c utils.o file_traverse.o ocl_worker.o \
	libacmatch.a ocl_prefix_sum.o ocl_compact_array.o
	$(CC) $(CCFLAGS) $^ $(LIBOCL) $(LIBTHREAD) $(LIBMATH) -o $@

libacmatch.a: ocl_context.o databuf.o ocl_aho_match.o acsmx.o
	ar rcs $@ $^

databuf_test: databuf.c utils.o ocl_context.o ocl_aho_match.o acsmx.o \
	ocl_prefix_sum.o ocl_compact_array.o
	$(CC) $(DBGFLAGS) -DDATABUF_TEST $^ $(LIBOCL) $(LIBMATH) -o $@

compact_array_test: utils.o ocl_context.o \
	ocl_prefix_sum.o ocl_compact_array.c
	$(CC) $(DBGFLAGS) -DCOMPACT_ARRAY_TEST $^ $(LIBOCL) $(LIBMATH) -o $@

clean:
	rm -f $(TARGETS) $(UNIT_TESTS) *.o

# header deps
ocl_aho_grep.o: utils.h ocl_context.h databuf.h
utils.o: utils.h common.h
ocl_context.o: ocl_context.h common.h
databuf.o: databuf.h common.h ocl_context.h
ocl_aho_match.o: ocl_aho_match.h acsmx.h ocl_context.h common.h file_traverse.h
ocl_prefix_sum.o: ocl_prefix_sum.c ocl_prefix_sum.h
ocl_compact_array.o: ocl_compact_array.c ocl_compact_array.h
ocl_worker.o: ocl_worker.h common.h ocl_context.h acsmx.h databuf.h utils.h
acsmx.o: acsmx.h common.h ocl_context.h
file_traverse.o: file_traverse.c file_traverse.h
