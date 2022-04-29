#include <sys/stat.h>

#include "ocl_compact_array.h"
#include "utils.h"

static void
ocl_compact_array_kernel(struct clconf *cl, cl_mem results_comp, cl_mem results,
    cl_mem prefixsum, cl_uint chunks, cl_int max_results, size_t local_ws);

extern char* strload(const char *);
static char* LoadProgramSourceFromFile(const char *filename);

#if 1
void
ocl_compact_array_init(struct clconf *cl) {
    int err;
    const char* filename = "./compactarray.cl";

    char *source = LoadProgramSourceFromFile(filename);
    if(!source) {
        ERRX(EXIT_FAILURE, "Error: Failed to load program from file!\n");
    }

    /* Create the compute program from the source buffer */
    cl->program_compact_array = clCreateProgramWithSource(cl->ctx, 1,
		    (const char **) & source, NULL, &err);
    if (!cl->program_compact_array || err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "Error: Failed to create compute program!\n");
    }
    
    /* Build the program executable */
    err = clBuildProgram(cl->program_compact_array, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t length;
        char build_log[2048];
        printf("%s\n", source);
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(cl->program_compact_array, cl->dev,
			CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log,
			&length);
        printf("%s\n", build_log);
        ERRX(EXIT_FAILURE, "");
    }

    free(source); source = NULL;

    /* load kernel */
    cl->kernel_compact_array = clCreateKernel(
		cl->program_compact_array, "compactarray", &err);
    if (!cl->kernel_prescan || err != CL_SUCCESS) {
            ERRX(EXIT_FAILURE, "Error: Failed to create compute kernel!\n");
    }

    return;
}

#else
void
ocl_compact_array_init(struct clconf *cl) {
	int e, ret = 0;
	const char *opts = NULL;
	char *optbuf;
	unsigned int optlen;
	const char *kstr = NULL;
	const char *kname = "compactarray";

	kstr = (const char*)strload("compactarray.cl");

	if (kstr == NULL)
		ERRX(1, "strload compactarray.cl");

	/* add cwd to include path to keep the amd sdk happy */
#define CWDINCSTR "-I./ "

	optlen = strlen(CWDINCSTR) + 1;
	if (opts != NULL)
		optlen += strlen(opts);

	optbuf = calloc(1, optlen);
	if (optbuf == NULL)
		ERRX(1, "malloc optbuf");

	strcpy(optbuf, CWDINCSTR);
	if (opts != NULL)
		strcpy(optbuf, opts);

	/* generate code */
	cl->program_compact_array = clCreateProgramWithSource(cl->ctx, 1, &kstr, NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR creating program_compact_array: %s", clstrerror(e));

	e = clBuildProgram(cl->program_compact_array, 0, NULL, optbuf, NULL, NULL);
	clputlog(cl);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR building program_compact_array: %s", clstrerror(e));

	cl->kernel_compact_array = clCreateKernel(cl->program_compact_array, kname, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR creating kernel_compact_array: %s",
				clstrerror(e));

	if (kstr) {
		free((char*)kstr);
		kstr = NULL;
	}

	return;
}
#endif

void
ocl_compact_array_close(struct clconf *c) {
	cl_int e;

	e  = clReleaseKernel(c->kernel_compact_array);
	e |= clReleaseProgram(c->program_compact_array);

	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR releasing OpenCL kernel and program: %s",
				clstrerror(e));

	return;
}


/*
 * OpenCL compact array kernel wrapper ( exposed )
 */
void
ocl_compact_array(struct clconf *cl, struct databuf *db, size_t local_ws)
{
	//TODO: XXX db->chunks must be > 0

	ocl_compact_array_kernel(cl, db->d_results_comp, db->d_results, db->d_prefixsum,
	    db->chunks, db->max_results, local_ws);

	ocl_compact_array_kernel(cl, db->d_results2_comp, db->d_results2, db->d_prefixsum,
	    db->chunks, db->max_results, local_ws);
}


/*
 * OpenCL compact array kernel wrapper
 */
static void
ocl_compact_array_kernel(struct clconf *cl, cl_mem results_comp, cl_mem results,
    cl_mem prefixsum, cl_uint chunks, cl_int max_results, size_t local_ws)
{
	int e;
	size_t global = ROUNDUP(chunks, local_ws);
	size_t local  = local_ws;

	/* Set the arguments */
	clSetKernelArg(cl->kernel_compact_array, 0, sizeof(cl_mem), &results_comp);
	clSetKernelArg(cl->kernel_compact_array, 1, sizeof(cl_mem), &results);
	clSetKernelArg(cl->kernel_compact_array, 2, sizeof(cl_mem), &prefixsum);
	clSetKernelArg(cl->kernel_compact_array, 3, sizeof(cl_int), &chunks);
	clSetKernelArg(cl->kernel_compact_array, 4, sizeof(cl_int), &max_results);

	/* execute the matching kernel */
	e = clEnqueueNDRangeKernel(cl->queue, cl->kernel_compact_array, 1, NULL, &global, &local, 0,
	    NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "kernel_compact_array: executing kernel: %s", clstrerror(e));

	clFlush(cl->queue);

	/* wait until the kernel is done */
	e = clFinish(cl->queue);
	if (e != CL_SUCCESS)
		ERRXV(1, "kernel_compact_array: finishing kernel: %s", clstrerror(e));
}

static char *
LoadProgramSourceFromFile(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

#ifdef COMPACT_ARRAY_TEST

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

	struct databuf *b;
	struct clconf cl;
	FILE *fp;
	unsigned char buf[128];
	char *line = NULL;
	unsigned char *chunk;
	ssize_t len = 0;
	int dev_pos = 0;
	unsigned int i, j;

	clinitctx(&cl, 0, -1);

	ocl_compact_array_init(&cl);

	return 0;
}
#endif

