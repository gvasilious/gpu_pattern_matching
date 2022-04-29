#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <CL/opencl.h>

#include "databuf.h"
#include "ocl_context.h"
#include "common.h"

int main(int argc, char **argv)
{
    int		i;
    uint64_t	t0 = 0;
    uint64_t	t1 = 0;
    uint64_t	t2 = 0;
    int		err = 0;
    cl_mem	output_buffer;
    cl_mem	input_buffer;
    int		count = 1024 * 1024;

    struct clconf cl;
    struct databuf db;

    struct timeval start, end;
    unsigned long time_total = 0;

    size_t buffer_size;
    float *float_data, *result;

    if (argc > 1) {
	    count = atoi(argv[1]);
    }

    buffer_size = sizeof(float) * count;

    /* Create some random input data on the host */
    float_data = (float*)malloc(count * sizeof(float));
    for (i = 0; i < count; i++)
    {
        float_data[i] = (int)(10 * ((float) rand() / (float) RAND_MAX));
    }

    /* Allocate and initialize the results array */
    result = (float*)malloc(buffer_size);
    memset(result, 0, buffer_size);

    /* initialize OpenCL context */
    clinitctx(&cl, 0, -1);

    /* initialize prefix sum */
    ocl_prefix_sum_init(&cl);

    /* Create the input buffer on the device */
    db.d_input = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    if (!db.d_input) {
        ERRX(EXIT_FAILURE, "create input buffer");
    }

    /* Create the output buffer on the device */
    db.d_output = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    if (!db.d_output) {
        ERRX(EXIT_FAILURE, "create output buffer");
    }

    /* Fill the input buffer with the host allocated random data */
    err = clEnqueueWriteBuffer(cl.queue, db.d_input, CL_TRUE, 0, buffer_size, float_data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "write input buffer");
    }

    /* Create the buffers for the partial sums */
    CreatePartialSumBuffers(&cl, &db, count);

    /* this is a warm-up */
    ocl_prefix_sum(&cl, &db, count);

    gettimeofday(&start, NULL);

    /* Do the actual run */
    ocl_prefix_sum(&cl, &db, count);

    err = clFinish(cl.queue);

    if (err != CL_SUCCESS) {
        ERRV(EXIT_FAILURE, "%d: clFinish", err);
    }

    gettimeofday(&end, NULL);

    /* Calculate the statistics for execution time and throughput */
    time_total += ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    printf("Scanning completed in %lu usec\n", time_total);
    printf("Scanning throughput: %.2f Mbit/sec\n", (buffer_size * 8.0) / time_total);

    /* Read back the results that were computed on the device */
    err = clEnqueueReadBuffer(cl.queue, db.d_output, CL_TRUE, 0, buffer_size, result, 0, NULL, NULL);
    if (err) {
        ERRX(EXIT_FAILURE, "read output buffer");
    }

    /* Shutdown and cleanup */
    ReleasePartialSums(&cl, &db);

    ocl_prefix_sum_close(&cl);

    clReleaseMemObject(db.d_input);
    clReleaseMemObject(db.d_output);
    
    free(float_data);
    free(result);
    
        
    return 0;
}

