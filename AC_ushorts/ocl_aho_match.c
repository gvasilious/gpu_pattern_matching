/*
 * authors:	Dimitris Deyannis
 * 		Eva Papadogiannaki
 * last update:	Mar-2017
 * 		Nov-2018
 * contact:	deyannis@ics.forth.gr
 * 		epapado@ics.forth.gr
 */

#include "ocl_aho_match.h"


/*
 * OpenCL Aho-Corasick match kernel wrapper
 */
void
ocl_aho_match_kernel(cl_kernel kernel, cl_command_queue queue, cl_mem trans,
    cl_mem data, cl_mem indices, cl_mem sizes, cl_mem results, cl_uint chunks,
    cl_ulong data_size, cl_long last_state, cl_int max_pat_size,
    cl_int max_results, size_t local_ws)
{
	int e;
	size_t global = ROUNDUP(chunks, local_ws);
	size_t local = local_ws;

	/* Set the arguments */
	clSetKernelArg(kernel, 0, sizeof(cl_mem),   &trans);
	clSetKernelArg(kernel, 1, sizeof(cl_mem),   &data);
	clSetKernelArg(kernel, 2, sizeof(cl_mem),   &indices);
	clSetKernelArg(kernel, 3, sizeof(cl_mem),   &sizes);
	clSetKernelArg(kernel, 4, sizeof(cl_mem),   &results);
	clSetKernelArg(kernel, 5, sizeof(cl_uint),  &chunks);
	clSetKernelArg(kernel, 6, sizeof(cl_ulong), &data_size);
	clSetKernelArg(kernel, 7, sizeof(cl_long),  &last_state);
	clSetKernelArg(kernel, 8, sizeof(cl_int),   &max_pat_size);
	clSetKernelArg(kernel, 9, sizeof(cl_int),   &max_results);

	/* execute the matching kernel */
	e = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0,
	    NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR executing kernel: %s", clstrerror(e));

	clFlush(queue);

	/* wait until the kernel is done */
	e = clFinish(queue);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR finishing kernel: %s", clstrerror(e));
}


/*
 * OpenCL Aho-Corasick match kernel wrapper ( exposed )
 */
void
ocl_aho_match(struct clconf *cl, struct databuf *db, iacsm_t *iacsm,
    size_t local_ws)
{
	ocl_aho_match_kernel(cl->kernel, cl->queue, iacsm->d_trans, db->d_data,
	    db->d_indices, db->d_sizes, db->d_results, db->chunks, db->bytes,
	    db->last_state, iacsm_get_max_pattern_size(iacsm), db->max_results,
	    local_ws);
}
