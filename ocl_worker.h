#ifndef _OCL_WORKER_H_
#define _OCL_WORKER_H_

#include "ocl_context.h"
#include "acsmx.h"
#include "databuf.h"


/* worker context */
struct ocl_worker_ctx {
	int            id;		/* context's thread id                */
	int            text_mode;	/* read input files line-wise         */
	int            follow;		/* output appended data as files grow */
	int            verbose;		/* context's verbosity flag           */
	int            thread_no;	/* total number of threads            */
	int            total_files;	/* total number of files              */
	int            *fds;		/* all file descriptors               */
	char           **filenames;	/* all file names                     */
	size_t         matches_total;	/* total matches in context           */
	size_t         matches_reported; /* matches reported                 */
	size_t         bytes;		/* total processed bytes in context   */
	size_t         lines;		/* total lines of text processed      */
	size_t         rounds;		/* total kernel calls in context      */
	size_t         global_ws;	/* context's global work size         */
	size_t         local_ws;	/* context's local work size          */
	struct clconf  cl;		/* context's OpenCL configuration     */
	struct databuf *db;		/* context's data buffer              */
	acsm_t         *acsm;		/* context's Aho-Corasick automaton   */
	acsm_pattern_t *patterns;	/* context's patterns                 */
	size_t         patterns_size;	/* total number of the patterns       */
};


/*
 * creates a new worker context
 *
 * arg0: device possition
 *
 * ret:  a new worker context
 *       NULL if the creation fails
 */
struct ocl_worker_ctx *
ocl_worker_ctx_create(int);


/*
 * initializes a new worker context
 *
 * arg00: worker context
 * arg01: device possition
 * arg02: local work size
 * arg03: global work size
 * arg04: mapped buffers flag
 * arg05: pattern file path
 * arg06: hex patterns flag
 * arg07: pattern size limit
 * arg08: maximum chunk size
 * arg09: maximum result cells per chunk
 * arg10: verbosity flag
 * arg11: text mode
 * arg12: follow
 * arg13: thread id
 * arg14: maximum number of cpu threads
 * arg15: total input files
 * arg16: file descriptors
 * arg17: file names
 *
 * ret:    0 if initialization was successful
 *        -1 if the initialization failed
 */
int
ocl_worker_ctx_init(struct ocl_worker_ctx *, int, size_t, size_t, int, char *,
    int, int, size_t, int, int, int, int, int, int, int, int *, char **);


/*
 * frees the worker context
 *
 * arg0: worker context 
 */
void
ocl_worker_ctx_free(struct ocl_worker_ctx *);


#endif /* _OCL_WORKER_H_ */
