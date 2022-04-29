/*
 * authors:	Dimitris Deyannis
 *		Eva Papadogiannaki
 * last update:	Mar-2017
 * 		Nov-2018
 * contact:	deyannis@ics.forth.gr
 * 		epapado@ics.forth.gr
 */

#ifndef _OCL_AHO_MATCH_H_
#define _OCL_AHO_MATCH_H_

#include "common.h"
#include "clutil.h"
#include "databuf.h"
#include "iacsmx.h"

#include <CL/opencl.h>


/*
 * OpenCL Aho-Corasick match kernel wrapper
 */
void
ocl_aho_match(struct clconf *, struct databuf *, iacsm_t *, size_t);


#endif /* _OCL_AHO_MATCH_H_ */
