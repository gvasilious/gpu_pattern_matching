/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <string.h>

#include "ocl_bitonic_sort.h"
#include "utils.h"


static const unsigned int LOCAL_SIZE_LIMIT = 512U;
static const char  *compileOptions = "-D LOCAL_SIZE_LIMIT=512";

int
ocl_bitonic_sort_init(struct clconf *cl_conf) {
	cl_int e;
	size_t kernelLength;
	int ret = 0;

	char *src = strload_ex("BitonicSort.cl", "// My comment\n", &kernelLength);
	if (src == NULL) {
		ret = -2;
		goto end;
	}

	cl_conf->bitonic_sort_program = clCreateProgramWithSource(cl_conf->ctx, 1, (const char **)&src, &kernelLength, &e);
	if (e != CL_SUCCESS) {
		ret = -1;
		goto end;
	}

	e = clBuildProgram(cl_conf->bitonic_sort_program, 0, NULL, compileOptions, NULL, NULL);
	if (e != CL_SUCCESS) {
		ret = -1;
		goto end;
	}

	/* Create kernels */
	cl_conf->bitonic_sort_local_kernel = clCreateKernel(cl_conf->bitonic_sort_program, "bitonicSortLocal", &e);
	if (e != CL_SUCCESS) {
		ret = -1;
		goto end;
	}

	cl_conf->bitonic_sort_local_kernel1 = clCreateKernel(cl_conf->bitonic_sort_program, "bitonicSortLocal1", &e);
	if (e != CL_SUCCESS) {
		ret = -1;
		goto end;
	}

	cl_conf->bitonic_merge_global_kernel = clCreateKernel(cl_conf->bitonic_sort_program, "bitonicMergeGlobal", &e);
	if (e != CL_SUCCESS) {
		ret = -1;
		goto end;
	}

	cl_conf->bitonic_merge_local_kernel = clCreateKernel(cl_conf->bitonic_sort_program, "bitonicMergeLocal", &e);
	if (e != CL_SUCCESS) {
		ret = -1;
		goto end;
	}

        /* Check for work group size */
        cl_device_id device;
        size_t sort_local_sz, sort_local_sz1, sort_merge_sz;

	e  = clGetCommandQueueInfo(cl_conf->queue,
			CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL);

	e |= clGetKernelWorkGroupInfo(cl_conf->bitonic_sort_local_kernel, device,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
			&sort_local_sz, NULL);

	e |= clGetKernelWorkGroupInfo(cl_conf->bitonic_sort_local_kernel1, device,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
			&sort_local_sz1, NULL);

	e |= clGetKernelWorkGroupInfo(cl_conf->bitonic_merge_local_kernel, device,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t),
			&sort_merge_sz, NULL);

	if (e != CL_SUCCESS) {
		ret = -1;
		goto end;
	}

	if( (e != CL_SUCCESS || sort_local_sz < (LOCAL_SIZE_LIMIT / 2))
	    || (sort_local_sz1 < (LOCAL_SIZE_LIMIT / 2))
	    || (sort_merge_sz < (LOCAL_SIZE_LIMIT / 2)) ) {

		ret = -1;
		goto end;
	}

end:
	if (src) { /* free temp resources */
		free(src);
		src = NULL;
	}

	if (ret < 0) { /* we have an error */
		ocl_bitonic_sort_close(cl_conf);
	}

	return ret;
}

int ocl_bitonic_sort_close(struct clconf *c) {
	cl_int e;

	e  = clReleaseKernel(c->bitonic_merge_local_kernel);
	e |= clReleaseKernel(c->bitonic_merge_global_kernel);
	e |= clReleaseKernel(c->bitonic_sort_local_kernel1);
	e |= clReleaseKernel(c->bitonic_sort_local_kernel);
	e |= clReleaseProgram(c->bitonic_sort_program);

	if (e != CL_SUCCESS) {
		return -1;
	}
	return 0;
}

static cl_uint factorRadix2(cl_uint *log2L, cl_uint L) {
	if(!L) {
		*log2L = 0;
		return 0;
	} else {
		for(*log2L = 0; (L & 1) == 0; L >>= 1, (*log2L)++);
		return L;
	}
}

int ocl_bitonic_sort(struct clconf *cl_conf, cl_mem d_key_dst, cl_mem d_val_dst,
		cl_mem d_key_src, cl_mem d_val_src, unsigned int batch,
		unsigned int len, unsigned int dir) {

	cl_int e;
	cl_uint log2L;
	size_t localWorkSize; 
	size_t globalWorkSize;

	/* too short to sort */
	if(len < 2)
		return 0;

    
	/* Only power-of-two array lengths are supported */
	cl_uint factorizationRemainder = factorRadix2(&log2L, len);
	if (factorizationRemainder != 1) {
		return -1;
	}

	dir = (dir != 0);
    
	if(len <= LOCAL_SIZE_LIMIT) {
		if (((batch * len) % LOCAL_SIZE_LIMIT) != 0)
			return -1;

		/* launch bitonicSortLocal */
		e  = clSetKernelArg(cl_conf->bitonic_sort_local_kernel, 0, sizeof(cl_mem), &d_key_dst);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel, 1, sizeof(cl_mem), &d_val_dst);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel, 2, sizeof(cl_mem), &d_key_src);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel, 3, sizeof(cl_mem), &d_val_src);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel, 4, sizeof(cl_uint), &len);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel, 5, sizeof(cl_uint), &dir);

		if (e != CL_SUCCESS) {
			return -1;
		}

		localWorkSize  = LOCAL_SIZE_LIMIT / 2;
		globalWorkSize = batch * len / 2;
		e = clEnqueueNDRangeKernel(cl_conf->queue, cl_conf->bitonic_sort_local_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		if (e != CL_SUCCESS) {
			return -1;
		}
	} else {
		/* launch bitonicSortLocal1 */
		e  = clSetKernelArg(cl_conf->bitonic_sort_local_kernel1, 0, sizeof(cl_mem), &d_key_dst);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel1, 1, sizeof(cl_mem), &d_val_dst);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel1, 2, sizeof(cl_mem), &d_key_src);
		e |= clSetKernelArg(cl_conf->bitonic_sort_local_kernel1, 3, sizeof(cl_mem), &d_val_src);
		if (e != CL_SUCCESS) {
			return -1;
		}

		localWorkSize = LOCAL_SIZE_LIMIT / 2;
		globalWorkSize = batch * len / 2;
		e = clEnqueueNDRangeKernel(cl_conf->queue, cl_conf->bitonic_sort_local_kernel1, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
		if (e != CL_SUCCESS) {
			return -1;
		}

		for(unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= len; size <<= 1) {
			for(unsigned stride = size / 2; stride > 0; stride >>= 1) {
				if(stride >= LOCAL_SIZE_LIMIT) {
					/* launch bitonicMergeGlobal */
					e  = clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 0, sizeof(cl_mem), &d_key_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 1, sizeof(cl_mem), &d_val_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 2, sizeof(cl_mem), &d_key_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 3, sizeof(cl_mem), &d_val_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 4, sizeof(cl_uint), &len);
					e |= clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 5, sizeof(cl_uint), &size);
					e |= clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 6, sizeof(cl_uint), &stride);
					e |= clSetKernelArg(cl_conf->bitonic_merge_global_kernel, 7, sizeof(cl_uint), &dir);
					if (e != CL_SUCCESS) {
						return -1;
					}

					globalWorkSize = batch * len / 2;
					e = clEnqueueNDRangeKernel(cl_conf->queue, cl_conf->bitonic_merge_global_kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
					if (e != CL_SUCCESS) {
						return -1;
					}
				}
				else {
					/* launch bitonicMergeLocal */
					e  = clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 0, sizeof(cl_mem), &d_key_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 1, sizeof(cl_mem), &d_val_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 2, sizeof(cl_mem), &d_key_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 3, sizeof(cl_mem), &d_val_dst);
					e |= clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 4, sizeof(cl_uint), &len);
					e |= clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 5, sizeof(cl_uint), &stride);
					e |= clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 6, sizeof(cl_uint), &size);
					e |= clSetKernelArg(cl_conf->bitonic_merge_local_kernel, 7, sizeof(cl_uint), &dir);
					if (e != CL_SUCCESS) {
						return -1;
					}

					localWorkSize  = LOCAL_SIZE_LIMIT / 2;
					globalWorkSize = batch * len / 2;

					e = clEnqueueNDRangeKernel(cl_conf->queue, cl_conf->bitonic_merge_local_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
					if (e != CL_SUCCESS) {
						return -1;
					}
					break;
				}
			}
		}
	}

	return localWorkSize;
}


