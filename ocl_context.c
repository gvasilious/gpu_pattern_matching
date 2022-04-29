#include <stdio.h>
#include <stdlib.h>

/* support for all APIs */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <CL/opencl.h>
#include <CL/cl_ext.h>

#include "common.h"
#include "ocl_context.h"
#include "utils.h"

/*
 * creates a new OpenCL context
 */
void
clinitctx(struct clconf *cl, int pos, int subpos)
{
	int e;
	int i;
	int k = 0; /* global index */
	int done = 0;
	cl_uint nplatforms;
	cl_platform_id *platform;

	/* get platforms */
	e = clGetPlatformIDs(0, NULL, &nplatforms);
	if (e != CL_SUCCESS)
		ERRXV(1, "nplatforms: %s", clstrerror(e));
	DPRINTF_U(nplatforms);
	platform = calloc(nplatforms, sizeof(cl_platform_id));
	e = clGetPlatformIDs(nplatforms, platform, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "platform: %s", clstrerror(e));

	/* for all platforms */
	for (i = 0; i < nplatforms; i++) {
		int j;
		cl_uint ndevs;
		cl_device_id *dev;

		/* get devices */
		e = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL,
		    0, NULL, &ndevs);
		if (e != CL_SUCCESS)
			ERRXV(1, "ndevs: %s", clstrerror(e));
		DPRINTF_U(ndevs);
		dev = calloc(ndevs, sizeof(cl_device_id));
		e = clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL,
		    ndevs, dev, NULL);
		if (e != CL_SUCCESS)
			ERRXV(1, "dev: %s", clstrerror(e));

		/* for all devices */
		for (j = 0; j < ndevs; j++) {
			if (k++ == pos) {
				cl->platform = platform[i];
				cl->dev = dev[j];
				done = 1;
			}
		}
		free(dev);
	}

	free(platform);

	if (!done)
		ERRX(1, "invalid dev pos");

	/* create environment */
	cl->ctx = clCreateContext(NULL, 1, &(cl->dev), NULL, NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ctx: %s", clstrerror(e));
	cl->queue = clCreateCommandQueue(cl->ctx, cl->dev, 0, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "queue: %s", clstrerror(e));
	e = clGetDeviceInfo(cl->dev, CL_DEVICE_TYPE,
	    sizeof(cl->type), &(cl->type), NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "type: %s", clstrerror(e));

	return;
}

