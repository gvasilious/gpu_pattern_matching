/*
 * author:	Dimitris Deyannis
 * last update:	Mar-2017
 * contact:	deyannis@ics.forth.gr
 */

#include <unistd.h>
#include <time.h>
#include <ctype.h>

#include "utils.h"
#include "common.h"


/*
 * converts half printable hex value to integer
 */
int
half_hex_to_int(unsigned char c)
{
	if (isdigit(c))
		return c - '0';

	if ((tolower(c) >= 'a') && (tolower(c) <= 'f'))
		return tolower(c) + 10 - 'a';
}


/*
 * converts a printable hex array to ASCII
 */
unsigned char *
printable_hex_to_bytes(unsigned char *input)
{
	int i;
	unsigned char *output;

	output = NULL;
	if (strlen(input) % 2 != 0) {
		printf("ERROR: reading pattern!\n");
		exit(EXIT_FAILURE);
	}

	output = calloc(strlen(input) / 2, sizeof(unsigned char));
	if (!output)
		ERRX(1, "ERROR: calloc output\n");

	for (i = 0; i < strlen(input); i+= 2) {
		output[i / 2] = ((unsigned char)half_hex_to_int(input[i])) *
		    16 + ((unsigned char)half_hex_to_int(input[i + 1]));
	}
	
	return output;
}


/*
 * returns the current time in usecs
 */
size_t
gettime(void)
{
	struct timespec tp;

	clock_gettime(CLOCK_MONOTONIC, &tp);

	return (tp.tv_sec * 1000000 + tp.tv_nsec / 1000.0);
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

	e = clGetProgramBuildInfo(cl->program_aho_match, cl->dev,
	    CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status,
	    NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "status: %s", clstrerror(e));

	if (status == CL_BUILD_SUCCESS)
		return;

	e = clGetProgramBuildInfo(cl->program_aho_match, cl->dev,
	    CL_PROGRAM_BUILD_LOG, 0, NULL, &logsiz);

	DPRINTF_U(logsiz);

	log = malloc(logsiz);
	clGetProgramBuildInfo(cl->program_aho_match, cl->dev,
	    CL_PROGRAM_BUILD_LOG, logsiz, log, NULL);

	fprintf(stderr, "%s\n", log);

	free(log); log = NULL;

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
 * loads the OpenCL source code and prepends the preamble to the code.
 */
char* strload_ex(const char* fname, const char* preamble, size_t* prog_len)
{
	FILE* fp = NULL;
	size_t code_len, preamble_len;
	char *src_str;

	fp = fopen(fname, "rb");
	if(fp == 0) {
		return NULL;
	}

	preamble_len = strlen(preamble);

	/* get the length of the source code */
	fseek(fp, 0, SEEK_END); 
	code_len = ftell(fp);
	fseek(fp, 0, SEEK_SET); 

	// allocate a buffer for the source code string and read it in
	src_str = (char *)malloc(code_len + preamble_len + 1);
	if (!src_str)
		return NULL;

	memcpy(src_str, preamble, preamble_len);
	if (fread((src_str) + preamble_len, code_len, 1, fp) != 1) {
		fclose(fp);
		free(src_str);
		return 0;
	}

	/* close the file */
	fclose(fp);

	/* return the total length of the combined (preamble + source) string*/
	if(prog_len != 0) {
		*prog_len = code_len + preamble_len;
	}

	src_str[code_len + preamble_len] = '\0';

	return src_str;
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
