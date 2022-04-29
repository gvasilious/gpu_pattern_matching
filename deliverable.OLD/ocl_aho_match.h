#ifndef _OCL_AHO_MATCH_H_
#define _OCL_AHO_MATCH_H_

#include "common.h"
#include "clutil.h"
#include "databuf.h"
#include "acsmx.h"

#include <CL/opencl.h>


/*
 * OpenCL Aho-Corasick match kernel wrapper
 */
void
ocl_aho_match(struct clconf *, struct databuf *, acsm_t *, size_t);


#endif /* _OCL_AHO_MATCH_H_ */
