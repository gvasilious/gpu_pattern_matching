#ifndef _OCL_BITONIC_SORT_
#define _OCL_BITONIC_SORT_

#include <stdio.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include "ocl_context.h"

int ocl_bitonic_sort_init(struct clconf*);

int ocl_bitonic_sort_close(struct clconf*);

int ocl_bitonic_sort(struct clconf*, cl_mem, cl_mem, cl_mem, cl_mem, unsigned int,
		unsigned int, unsigned int dir);

#endif /* _OCL_BITONIC_SORT_ */
