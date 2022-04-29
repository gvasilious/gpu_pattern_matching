#ifndef _CLUTIL_H_
#define _CLUTIL_H_

#include <CL/opencl.h>


/* OpenCL configuration */
struct clconf {
	cl_platform_id		platform;	/* Used platform        */
	cl_device_id 		dev;		/* Used device id       */
	cl_context 		ctx;		/* OpenCL context       */
	cl_command_queue	queue;		/* OpenCL command queue */
	cl_program 		program;	/* OpenCL program       */
	cl_kernel 		kernel;		/* OpenCL kernel        */
	cl_device_type 		type;		/* Used device type     */
};


/*
 * creates a new OpenCL context and builds the kernel from file
 *
 * arg0: OpenCL configuration
 * arg1: path to file containing the kernel source
 * arg2: kernel name
 * arg3: build options
 * arg4: devise position
 * arg5: device sub-position
 */
void
clsetupf(struct clconf *, const char *, const char *, const char *, int, int);


/*
 * creates a new OpenCL context
 *
 * arg0: OpenCL configuration
 * arg1: device position
 * arg2: device subposition
 */
void
clinitctx(struct clconf *, int, int);


/*
 * builds the OpenCL kernel
 *
 * arg0: OpenCL configuration
 * arg1: string containing the OpenCL kernel source
 * arg2: kernel name
 * arg3: build options
 */
void
clinitcode(struct clconf *, const char *, const char *, const char *);


/*
 * builds the OpenCL kernel from file
 *
 * arg0: OpenCL configuration
 * arg1: path to file containing the kernel source
 * arg2: kernel name
 * arg3: build options
 */
void
clinitcodef(struct clconf *, const char *, const char *, const char *);


/*
 * prints the OpenCL build log
 *
 * arg0: OpenCL configuration
 */
void
clputlog(struct clconf *);


/*
 * OpenCL error code to string
 *
 * arg0: OpenCL error code
 *
 * ret:  A string explaining the OpenCL error code
 */
char *
clstrerror(int);


#endif /* _CLUTIL_H_ */
