CC = gcc -O2
CCFLAGS = -Wall -pedantic
DBGFLAGS = -g
LIBOCL = -lOpenCL
LIBTHREAD = -lpthread

TARGETS = ocl_aho_grep 

all: $(TARGETS)

ocl_aho_grep: ocl_aho_grep.c utils.o file_traverse.o ocl_worker.o \
	libacmatch.a
	$(CC) $(CCFLAGS) $^ $(LIBOCL) $(LIBTHREAD) -o $@

clean:
	rm -f $(TARGETS) *.o

# header deps
ocl_aho_grep.o: utils.h
utils.o: utils.h common.h
ocl_worker.o: ocl_worker.h common.h utils.h
file_traverse.o: file_traverse.c file_traverse.h
