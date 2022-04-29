#ifndef _OCL_PREFIX_SUM_H_
#define _OCL_PREFIX_SUM_H_

#include "common.h"
#include "ocl_context.h"
#include "databuf.h"
#include "acsmx.h"

#include <CL/opencl.h>


void
ocl_prefix_sum_init(struct clconf *c);

void
ocl_prefix_sum_close(struct clconf *c);

/*
 * OpenCL Prefix sum kernel wrapper
 */
void
ocl_prefix_sum(struct clconf *, struct databuf *, unsigned int);


#endif /* _OCL_PREFIX_SUM_H_ */
