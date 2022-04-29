#include "ocl_aho_match.h"
#include "utils.h"

static void
ocl_aho_match_kernel(struct clconf *cl, cl_mem trans,
    cl_mem data, cl_mem indices, cl_mem sizes, cl_mem results, cl_mem results2,
    cl_uint chunks, cl_ulong data_size, cl_long last_state, cl_int max_pat_size,
    cl_int max_results, size_t local_ws, int stream);

extern char* strload(const char *);

void
ocl_aho_match_init(struct clconf *cl) {
	int e, ret = 0;
	const char *opts = NULL;
	char *optbuf;
	unsigned int optlen;
	const char *kstr = NULL;
	const char *kname = "ahomatch";

	kstr = (const char*)strload("ahomatch.cl");

	if (kstr == NULL)
		ERRX(1, "strload ahomatch.cl");

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
	cl->program_aho_match = clCreateProgramWithSource(cl->ctx, 1, &kstr, NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR creating OpenCL program: %s", clstrerror(e));

	e = clBuildProgram(cl->program_aho_match, 0, NULL, optbuf, NULL, NULL);
	clputlog(cl);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR building OpenCL program: %s", clstrerror(e));

	cl->kernel_aho_match = clCreateKernel(cl->program_aho_match, kname, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR creating OpenCL kernel: %s",
				clstrerror(e));

	if (kstr) {
		free((char*)kstr);
		kstr = NULL;
	}

	return;
}

void
ocl_aho_match_close(struct clconf *c) {
	cl_int e;

	e  = clReleaseKernel(c->kernel_aho_match);
	e |= clReleaseProgram(c->program_aho_match);

	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR releasing OpenCL kernel and program: %s",
				clstrerror(e));

	return;
}


/*
 * OpenCL Aho-Corasick match kernel wrapper ( exposed )
 */
void
ocl_aho_match(struct clconf *cl, struct databuf *db, acsm_t *acsm,
    size_t local_ws, int stream)
{
	ocl_aho_match_kernel(cl, acsm->d_trans, db->d_data,
	    db->d_indices, db->d_sizes, db->d_results, db->d_results2, db->chunks,
	    db->bytes, db->last_state, acsm_get_max_pattern_size(acsm),
	    db->max_results, local_ws, stream);
}


/*
 * OpenCL Aho-Corasick match kernel wrapper
 */
static void
ocl_aho_match_kernel(struct clconf *cl, cl_mem trans,
    cl_mem data, cl_mem indices, cl_mem sizes, cl_mem results, cl_mem results2,
    cl_uint chunks, cl_ulong data_size, cl_long last_state, cl_int max_pat_size,
    cl_int max_results, size_t local_ws, int stream)
{
	int e;
	size_t global = ROUNDUP(chunks, local_ws);
	size_t local = local_ws;

	/* Set the arguments */
	clSetKernelArg(cl->kernel_aho_match, 0, sizeof(cl_mem),   &trans);
	clSetKernelArg(cl->kernel_aho_match, 1, sizeof(cl_mem),   &data);
	clSetKernelArg(cl->kernel_aho_match, 2, sizeof(cl_mem),   &indices);
	clSetKernelArg(cl->kernel_aho_match, 3, sizeof(cl_mem),   &sizes);
	clSetKernelArg(cl->kernel_aho_match, 4, sizeof(cl_mem),   &results);
	clSetKernelArg(cl->kernel_aho_match, 5, sizeof(cl_mem),   &results2);
	clSetKernelArg(cl->kernel_aho_match, 6, sizeof(cl_uint),  &chunks);
	clSetKernelArg(cl->kernel_aho_match, 7, sizeof(cl_ulong), &data_size);
	clSetKernelArg(cl->kernel_aho_match, 8, sizeof(cl_long),  &last_state);
	clSetKernelArg(cl->kernel_aho_match, 9, sizeof(cl_int),   &max_pat_size);
	clSetKernelArg(cl->kernel_aho_match, 10, sizeof(cl_int),   &max_results);

	/* execute the matching kernel */
	e = clEnqueueNDRangeKernel(cl->queue, cl->kernel_aho_match, 1, NULL, &global, &local, 0,
	    NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ocl_aho_match_kernel: ERROR executing kernel: %s", clstrerror(e));

	clFlush(cl->queue);

	/* wait until the kernel is done */
	e = clFinish(cl->queue);
	if (e != CL_SUCCESS)
		ERRXV(1, "ocl_aho_match_kernel: ERROR finishing kernel: %s", clstrerror(e));
}

