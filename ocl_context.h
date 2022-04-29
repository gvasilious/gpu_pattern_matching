#ifndef _OCL_CONTEXT_H_
#define _OCL_CONTEXT_H_

#include <CL/opencl.h>


/* OpenCL configuration */
struct clconf {
	cl_platform_id   platform;		/* Used platform            */
	cl_device_id     dev;			/* Used device id           */
	cl_context       ctx;			/* OpenCL context           */
	cl_command_queue queue;			/* OpenCL command queue     */

	cl_program       program_aho_match;	/* OpenCL matching program  */
	cl_kernel        kernel_aho_match;	/* OpenCL matching kernel   */

	cl_program       program_prefixsum;	/* OpenCL prefixsum program */
	cl_kernel        kernel_prescan;
	cl_kernel        kernel_prescan_store_sum;
	cl_kernel        kernel_prescan_store_sum_non_power_of_two;
	cl_kernel        kernel_prescan_non_power_of_two;
	cl_kernel        kernel_uniform_add;

	cl_program       program_compact_array; /* OpenCL compaction program*/
	cl_kernel        kernel_compact_array;  /* OpenCL compaction kernel */

	cl_device_type   type;			/* Used device type         */
};

/*
 * creates a new OpenCL context
 *
 * arg0: OpenCL configuration
 * arg1: device position
 * arg2: device subposition
 */
void
clinitctx(struct clconf *, int, int);

#endif /* _OCL_CONTEXT_H_ */
