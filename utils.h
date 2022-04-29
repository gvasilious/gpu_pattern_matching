/*
 * author:	Dimitris Deyannis
 * last update:	Mar-2017
 * contact:	deyannis@ics.forth.gr
 */

#ifndef _UTILS_H_
#define _UTILS_H_

#include "ocl_worker.h"


/* maximum pattern size in bytes */
#define MAX_PAT_SIZE 4096


/*
 * prints the usage message
 */
void
usage(void);


/*
 * checks the user's arguments
 *
 * arg0: pattern file path
 * arg1: input file path
 * arg2: device possition
 * arg3: global work size
 * arg4: local work size
 * arg5: maximum chunk size
 * arg6: number of threads
 * arg7: pattern size limit in Bytes
 * arg8: maximum result cells per chunk
 */
void
check_args(char *, char *, int, size_t, size_t, size_t, int, int, int);


/*
 * checks if the input parameters are aligned
 * changes those who are not
 * arg0: local work size
 * arg1: global work size
 * arg2: maximum chunk size
 *
 * ret:  number of changed parameters
 */
int
align_parameters(size_t *, size_t *, size_t *);


/*
 * prints which file matched which pattern for the given chunk
 *
 * arg0: worker context
 * arg1: chunk id
 */
void
print_matches(struct ocl_worker_ctx *, int);


/*
 * returns the current time in usecs
 *
 * ret: current time in usecs
 */
size_t
gettime(void);


/*
 * converts a printable hex array to ASCII
 *
 * arg0: string containing the pattern in hex format
 *
 * ret:  string containing the pattern in ASCII
 */
unsigned char *
printable_hex_to_bytes(unsigned char *);

/*
 * prints the OpenCL build log
 *
 * arg0: OpenCL configuration
 */
void
clputlog(struct clconf *);


/*
 * OpenCL error code to string
 *
 * arg0: OpenCL error code
 *
 * ret:  A string explaining the OpenCL error code
 */
char *
clstrerror(int);

/*
 * loads the OpenCL source code
 *
 * arg0: the path of the OpenCL source code file
 */
char *
strload(const char*);

/*
 * loads the OpenCL source code and prepends the preamble to the code.
 *
 * arg0: the path of the OpenCL source code file
 * arg1: a preamble to be added to the source code
 * arg2: the final size of the program
 */
char*
strload_ex(const char*, const char*, size_t*);

#endif /* _UTILS_H_ */
