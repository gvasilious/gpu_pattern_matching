#ifndef _OCL_COMPACT_ARRAY_H_
#define _OCL_COMPACT_ARRAY_H_

#include "common.h"
#include "ocl_context.h"
#include "databuf.h"
#include "acsmx.h"

#include <CL/opencl.h>


void
ocl_compact_array_init(struct clconf *c);

void
ocl_compact_array_close(struct clconf *c);

/*
 * OpenCL Prefix sum kernel wrapper
 */
void
ocl_compact_array(struct clconf *, struct databuf *, size_t);


#endif /* _OCL_COMPACT_ARRAY_H_ */
