#ifndef _OCL_AHO_MATCH_H_
#define _OCL_AHO_MATCH_H_

#include "common.h"
#include "ocl_context.h"
#include "databuf.h"
#include "acsmx.h"

#include <CL/opencl.h>


void
ocl_aho_match_init(struct clconf *c);

void
ocl_aho_match_close(struct clconf *c);

/*
 * OpenCL Aho-Corasick match kernel wrapper
 *
 * @arg0: OpenCL configuration
 * @arg1: databuf to search
 * @arg2: the aho-corasick state machine
 * @arg3: local work size
 * @arg4: stream mode: continue to
 *        consecutive chunks if on; otherwise
 *        match each chunk independently
 */
void
ocl_aho_match(struct clconf *, struct databuf *, acsm_t *, size_t, int);


#endif /* _OCL_AHO_MATCH_H_ */
