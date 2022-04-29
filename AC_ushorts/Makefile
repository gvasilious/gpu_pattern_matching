CC = gcc #-O2
CCFLAGS = -Wall -pedantic
#CPPFLAGS += -D_EMPTY
#DBGFLAGS = -g
LIBOCL = -lOpenCL
LIBTHREAD = -lpthread

TARGETS = libiacmatch.a ocl_aho_grep  

all: $(TARGETS)

ocl_aho_grep: ocl_aho_grep.c utils.o file_traverse.o ocl_worker.o nmutil.o \
	libiacmatch.a
	$(CC) $(DBGFLAGS) $(CCFLAGS) $(CPPFLAGS) $^ $(LIBOCL) $(LIBTHREAD) -o $@

libiacmatch.a: clutil.o nmutil.o databuf.o ocl_aho_match.o iacsmx.o 
	ar rcs $@ $^

clean:
	rm -f $(TARGETS) *.o

# header deps
ocl_aho_grep.o: utils.h clutil.h databuf.h
utils.o: utils.h common.h
clutil.o: clutil.h common.h
databuf.o: databuf.h common.h clutil.h
ocl_aho_match.o: ocl_aho_match.h iacsmx.h clutil.h common.h file_traverse.h
ocl_worker.o: ocl_worker.h common.h clutil.h iacsmx.h databuf.h utils.h
iacsmx.o: iacsmx.h common.h clutil.h
file_traverse.o: file_traverse.h
nmutil.o: nmutil.h common.h	
