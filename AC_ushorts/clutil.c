#include <stdio.h>
#include <stdlib.h>

/* support for all APIs */
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <CL/opencl.h>
#include <CL/cl_ext.h>

#include "common.h"
#include "clutil.h"


/*
 * loads the OpenCL source code
 */
char *
strload(const char *path);


/*
 * creates a new OpenCL context and builds the kernel from file
 */
void
clsetupf(struct clconf *cl, const char *kpath, const char *kname,
    const char *opts, int pos, int subpos)
{
	clinitctx(cl, pos, subpos);
	clinitcodef(cl, kpath, kname, opts);
}


/*
 * builds the OpenCL kernel from file
 */
void
clinitcodef(struct clconf *cl, const char *kpath, const char *kname,
    const char *opts)
{
	char *kstr;

	kstr = strload(kpath);
	DPRINTF_S(kstr);
	clinitcode(cl, kstr, kname, opts);
	free(kstr);
}


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


/*
 * builds the OpenCL kernel
 */
void
clinitcode(struct clconf *cl, const char *kstr, const char *kname,
    const char *opts)
{
	int e;
	char *optbuf;
	unsigned int optlen;

	/* add cwd to include path to keep the amd sdk happy */
#define CWDINCSTR "-I./ "
	optlen = strlen(CWDINCSTR) + 1;
	if (opts != NULL)
		optlen += strlen(opts);
	optbuf = calloc(1, optlen);
	if (optbuf == NULL)
		ERRX(1, "malloc optbuf");
	strcpy(optbuf, CWDINCSTR);
	if (opts != NULL)
		strcpy(optbuf, opts);

	DPRINTF_S(opts);

	/* generate code */
	cl->program = clCreateProgramWithSource(cl->ctx, 1,
	    &kstr, NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "program: %s", clstrerror(e));
	e = clBuildProgram(cl->program, 0, NULL, optbuf, NULL, NULL);
	clputlog(cl);
	if (e != CL_SUCCESS)
		ERRXV(1, "build: %s", clstrerror(e));
	cl->kernel = clCreateKernel(cl->program, kname, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "kernel: %s", clstrerror(e));

	return;
}


/*
 * prints the OpenCL build log
 */
void
clputlog(struct clconf *cl)
{
	int e;
	cl_build_status status;
	size_t logsiz;
	char *log;

	e = clGetProgramBuildInfo(cl->program, cl->dev,
	    CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status,
	    NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "status: %s", clstrerror(e));

	if (status == CL_BUILD_SUCCESS)
		return;

	e = clGetProgramBuildInfo(cl->program, cl->dev,
	    CL_PROGRAM_BUILD_LOG, 0, NULL, &logsiz);

	DPRINTF_U(logsiz);

	log = malloc(logsiz);
	clGetProgramBuildInfo(cl->program, cl->dev,
	    CL_PROGRAM_BUILD_LOG, logsiz, log, NULL);

	fprintf(stderr, "%s\n", log);

	return;
}


/*
 * loads the OpenCL source code
 */
char *
strload(const char *path)
{
	FILE *fp;
	char *str;
	size_t len;

	/* find out the size */
	fp = fopen(path, "r");
	if (fp == NULL)
		ERRXV(1, "fopen: %s", path);
	fseek(fp, 0, SEEK_END);
	len = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	/* allocate and load */
	str = calloc(len + 1, sizeof(char));
	if (fread(str, sizeof(char), len, fp) != len)
		ERRX(1, "fread");
	str[len] = '\0'; /* terminate */

	fclose(fp);

	return str;
}


/*
 * OpenCL error code to string
 */
char *
clstrerror(int err)
{
	static char ebuf[64];

	switch (err) {
	case CL_SUCCESS:
		return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND:
		return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE:
		return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE:
		return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:
		return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE:
		return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP:
		return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH:
		return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:
		return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE:
		return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE:
		return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET:
		return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
		return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE:
		return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM:
		return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE:
		return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES:
		return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE:
		return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR:
		return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT:
		return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
		return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE:
		return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER:
		return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY:
		return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS:
		return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM:
		return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE:
		return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME:
		return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION:
		return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL:
		return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX:
		return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE:
		return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE:
		return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS:
		return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION:
		return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE:
		return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE:
		return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET:
		return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST:
		return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT:
		return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION:
		return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT:
		return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE:
		return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL:
		return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE:
		return "CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY:
		return "CL_INVALID_PROPERTY";
	default:
		snprintf(ebuf, sizeof ebuf, "Unknown error: code %d", err);
		return ebuf;
	}
}
