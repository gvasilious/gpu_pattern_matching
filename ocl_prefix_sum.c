#include <math.h>
#include <sys/stat.h>

#include "common.h"
#include "ocl_prefix_sum.h"
#include "utils.h"

#define IS_POWER_OF_TWO(n)	(((n&(n-1))==0))

#define GROUP_SIZE	256

#define NUM_BANKS       (16)
#define MAX_ERROR       (1e-7)

static int PreScan( struct clconf *cl, size_t *global, size_t *local,
		size_t shared, cl_mem output_data, cl_mem input_data,
		unsigned int n, int group_index, int base_index);

static int PreScanStoreSum(struct clconf *cl, size_t *global, size_t *local, 
		size_t shared, cl_mem output_data, cl_mem input_data, 
		cl_mem partial_sums, unsigned int n, int group_index, 
		int base_index);

static int PreScanStoreSumNonPowerOfTwo(struct clconf *cl, size_t *global,
	        size_t *local, size_t shared, cl_mem output_data, 
		cl_mem input_data, cl_mem partial_sums, unsigned int n,
		int group_index, int base_index);

static int PreScanNonPowerOfTwo(struct clconf *cl, size_t *global,
		size_t *local, size_t shared, cl_mem output_data,
		cl_mem input_data, unsigned int n, int group_index,
		int base_index);

static int UniformAdd(struct clconf *cl, size_t *global, size_t *local,
		cl_mem output_data, cl_mem partial_sums, unsigned int n,
		unsigned int group_offset, unsigned int base_index);

static int PreScanBufferRecursive(struct clconf *cl, struct databuf *db,
		cl_mem output_data, cl_mem input_data,  int max_group_size,
		int max_work_item_count, int element_count, int level);


static char *
LoadProgramSourceFromFile(const char *filename)
{
    struct stat statbuf;
    FILE        *fh;
    char        *source;

    fh = fopen(filename, "r");
    if (fh == 0)
        return 0;

    stat(filename, &statbuf);
    source = (char *) malloc(statbuf.st_size + 1);
    fread(source, statbuf.st_size, 1, fh);
    source[statbuf.st_size] = '\0';

    return source;
}

static int floorPow2(int n)
{
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
}

void
ocl_prefix_sum_init(struct clconf *cl) {
    int err;
    const char* filename = "./scan_kernel.cl";

    char *source = LoadProgramSourceFromFile(filename);
    if(!source) {
        ERRX(EXIT_FAILURE, "Error: Failed to load program from file!\n");
    }

    /* Create the compute program from the source buffer */
    cl->program_prefixsum = clCreateProgramWithSource(cl->ctx, 1,
		    (const char **) & source, NULL, &err);
    if (!cl->program_prefixsum || err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "Error: Failed to create compute program!\n");
    }
    
    /* Build the program executable */
    err = clBuildProgram(cl->program_prefixsum, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t length;
        char build_log[2048];
        printf("%s\n", source);
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(cl->program_prefixsum, cl->dev,
			CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log,
			&length);
        printf("%s\n", build_log);
        ERRX(EXIT_FAILURE, "");
    }

    free(source); source = NULL;

    /* load kernels */

    cl->kernel_prescan = clCreateKernel(
		cl->program_prefixsum, "PreScanKernel", &err);
    if (!cl->kernel_prescan || err != CL_SUCCESS) {
            ERRX(EXIT_FAILURE, "Error: Failed to create compute kernel!\n");
    }

    cl->kernel_prescan_store_sum = clCreateKernel(
		cl->program_prefixsum, "PreScanStoreSumKernel", &err);
    if (!cl->kernel_prescan_store_sum || err != CL_SUCCESS) {
            ERRX(EXIT_FAILURE, "Error: Failed to create compute kernel!\n");
    }

    cl->kernel_prescan_store_sum_non_power_of_two = clCreateKernel(
		cl->program_prefixsum, "PreScanStoreSumNonPowerOfTwoKernel",
		&err);
    if (!cl->kernel_prescan_store_sum_non_power_of_two || err != CL_SUCCESS) {
            ERRX(EXIT_FAILURE, "Error: Failed to create compute kernel!\n");
    }

    cl->kernel_prescan_non_power_of_two = clCreateKernel(
		cl->program_prefixsum, "PreScanNonPowerOfTwoKernel", &err);
    if (!cl->kernel_prescan_non_power_of_two || err != CL_SUCCESS) {
            ERRX(EXIT_FAILURE, "Error: Failed to create compute kernel!\n");
    }

    cl->kernel_uniform_add = clCreateKernel(
		cl->program_prefixsum, "UniformAddKernel", &err);
    if (!cl->kernel_uniform_add || err != CL_SUCCESS) {
	    ERRX(EXIT_FAILURE, "Error: Failed to create compute kernel!\n");
    }

    return;
}

void
ocl_prefix_sum_close(struct clconf *cl) {

	cl_int e;

    e  = clReleaseKernel(cl->kernel_prescan);
    e |= clReleaseKernel(cl->kernel_prescan_store_sum);
    e |= clReleaseKernel(cl->kernel_prescan_store_sum_non_power_of_two);
    e |= clReleaseKernel(cl->kernel_prescan_non_power_of_two);
    e |= clReleaseKernel(cl->kernel_uniform_add);

    e |= clReleaseProgram(cl->program_prefixsum);
    e |= clReleaseCommandQueue(cl->queue);
    e |= clReleaseContext(cl->ctx);

	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR releasing OpenCL kernel and program: %s",
				clstrerror(e));

	return;
}


/*
 * OpenCL Prefix sum kernel wrapper ( exposed )
 */
void
ocl_prefix_sum(struct clconf *cl, struct databuf *db, unsigned int element_count)
{
    unsigned int max_group_size;
    unsigned int max_work_item_count;
    int err;
    size_t wgSize = 0;
    int groupSize;

    //TODO: XXX  element_count must be > 0

    size_t returned_size = 0;
    size_t max_workgroup_size = 0;
    err = clGetDeviceInfo(cl->dev, CL_DEVICE_MAX_WORK_GROUP_SIZE,
		    sizeof(size_t), &max_workgroup_size, &returned_size);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "Device info");
    }
	
    groupSize = min(GROUP_SIZE, max_workgroup_size);

    err = clGetKernelWorkGroupInfo(
		    cl->kernel_prescan, cl->dev,
		    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL); 
    groupSize = min(groupSize, wgSize);

    err |= clGetKernelWorkGroupInfo(
		    cl->kernel_prescan_store_sum, cl->dev,
		    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL); 
    groupSize = min(groupSize, wgSize);

    err |= clGetKernelWorkGroupInfo(
		    cl->kernel_prescan_store_sum_non_power_of_two, cl->dev,
		    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL); 
    groupSize = min(groupSize, wgSize);

    err |= clGetKernelWorkGroupInfo(
		    cl->kernel_prescan_non_power_of_two, cl->dev,
		    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL); 
    groupSize = min( GROUP_SIZE, wgSize );

    err |= clGetKernelWorkGroupInfo(
		    cl->kernel_uniform_add, cl->dev,
		    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL); 
    groupSize = min(groupSize, wgSize);

    if (err) {
	    ERRX(EXIT_FAILURE, "Error: Failed to get kernel work group size");
    }

    max_group_size = groupSize;
    max_work_item_count = groupSize;

    /* XXX use proper pointers if mapped or pinned */
    PreScanBufferRecursive(cl, db, db->d_prefixsum, db->d_results,
		    max_group_size, max_work_item_count, element_count, 0);

}


static int
PreScan(
    struct clconf *cl,
    size_t *global, 
    size_t *local, 
    size_t shared, 
    cl_mem output_data, 
    cl_mem input_data, 
    unsigned int n,
    int group_index, 
    int base_index)
{
    int err = CL_SUCCESS;
    err |= clSetKernelArg(cl->kernel_prescan, 0, sizeof(cl_mem), &output_data);
    err |= clSetKernelArg(cl->kernel_prescan, 1, sizeof(cl_mem), &input_data);
    err |= clSetKernelArg(cl->kernel_prescan, 2, shared,         0);
    err |= clSetKernelArg(cl->kernel_prescan, 3, sizeof(cl_int), &group_index);
    err |= clSetKernelArg(cl->kernel_prescan, 4, sizeof(cl_int), &base_index);
    err |= clSetKernelArg(cl->kernel_prescan, 5, sizeof(cl_int), &n);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "kernel_prescan: set kernel arguments");
    }

    err = CL_SUCCESS;
    err |= clEnqueueNDRangeKernel(cl->queue, cl->kernel_prescan, 1, NULL,
		    global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "kernel_prescan: execute kernel");
    }

    return CL_SUCCESS;
}

static int
PreScanStoreSum(
    struct clconf *cl,
    size_t *global, 
    size_t *local, 
    size_t shared, 
    cl_mem output_data, 
    cl_mem input_data, 
    cl_mem partial_sums,
    unsigned int n,
    int group_index, 
    int base_index)
{
    int err = CL_SUCCESS;
    err |= clSetKernelArg(cl->kernel_prescan_store_sum, 0, sizeof(cl_mem), &output_data);  
    err |= clSetKernelArg(cl->kernel_prescan_store_sum, 1, sizeof(cl_mem), &input_data);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum, 2, sizeof(cl_mem), &partial_sums);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum, 3, shared,         0);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum, 4, sizeof(cl_int), &group_index);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum, 5, sizeof(cl_int), &base_index);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum, 6, sizeof(cl_int), &n);
    if (err != CL_SUCCESS) {
	ERRX(EXIT_FAILURE, "prescan_store_sum: set kernel arguments");
    }

    err = CL_SUCCESS;
    err |= clEnqueueNDRangeKernel(cl->queue, cl->kernel_prescan_store_sum, 1, NULL, global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "prescan_store_sum: execute kernel");
    }
    
    return CL_SUCCESS;
}

static int
PreScanStoreSumNonPowerOfTwo(
    struct clconf *cl,
    size_t *global, 
    size_t *local, 
    size_t shared, 
    cl_mem output_data, 
    cl_mem input_data, 
    cl_mem partial_sums,
    unsigned int n,
    int group_index, 
    int base_index)
{

    int err = CL_SUCCESS;
    err |= clSetKernelArg(cl->kernel_prescan_store_sum_non_power_of_two, 0, sizeof(cl_mem), &output_data);  
    err |= clSetKernelArg(cl->kernel_prescan_store_sum_non_power_of_two, 1, sizeof(cl_mem), &input_data);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum_non_power_of_two, 2, sizeof(cl_mem), &partial_sums);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum_non_power_of_two, 3, shared,         0);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum_non_power_of_two, 4, sizeof(cl_int), &group_index);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum_non_power_of_two, 5, sizeof(cl_int), &base_index);
    err |= clSetKernelArg(cl->kernel_prescan_store_sum_non_power_of_two, 6, sizeof(cl_int), &n);
    if (err != CL_SUCCESS)
    {
        ERRX(EXIT_FAILURE, "kernel_prescan_store_sum_non_power_of_two: set kernel arguments");
    }

    err = CL_SUCCESS;
    err |= clEnqueueNDRangeKernel(cl->queue, cl->kernel_prescan_store_sum_non_power_of_two, 1, NULL, global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "kernel_prescan_store_sum_non_power_of_two: execute kernel");
    }

    return CL_SUCCESS;
}

static int
PreScanNonPowerOfTwo(struct clconf *cl, size_t *global, size_t *local,
		size_t shared, cl_mem output_data, cl_mem input_data, 
		unsigned int n, int group_index, int base_index) {
    unsigned int a = 0;
    int err = CL_SUCCESS;

    err |= clSetKernelArg(cl->kernel_prescan_non_power_of_two, 0, sizeof(cl_mem), &output_data);  
    err |= clSetKernelArg(cl->kernel_prescan_non_power_of_two, 1, sizeof(cl_mem), &input_data);
    err |= clSetKernelArg(cl->kernel_prescan_non_power_of_two, 2, shared,         0);
    err |= clSetKernelArg(cl->kernel_prescan_non_power_of_two, 3, sizeof(cl_int), &group_index);
    err |= clSetKernelArg(cl->kernel_prescan_non_power_of_two, 4, sizeof(cl_int), &base_index);
    err |= clSetKernelArg(cl->kernel_prescan_non_power_of_two, 5, sizeof(cl_int), &n);

    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "kernel_prescan_non_power_of_two: set kernel arguments");
    }

    err = CL_SUCCESS;

    err |= clEnqueueNDRangeKernel(cl->queue, cl->kernel_prescan_non_power_of_two, 1, NULL, global, local, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "kernel_prescan_non_power_of_two: execute kernel");
    }

    return CL_SUCCESS;
}

static int UniformAdd(
    struct clconf *cl,
    size_t *global, 
    size_t *local, 
    cl_mem output_data, 
    cl_mem partial_sums, 
    unsigned int n, 
    unsigned int group_offset, 
    unsigned int base_index) {

    unsigned int a = 0;

    int err = CL_SUCCESS;

    err |= clSetKernelArg(cl->kernel_uniform_add, 0, sizeof(cl_mem), &output_data);  
    err |= clSetKernelArg(cl->kernel_uniform_add, 1, sizeof(cl_mem), &partial_sums);
    err |= clSetKernelArg(cl->kernel_uniform_add, 2, sizeof(float),  0);
    err |= clSetKernelArg(cl->kernel_uniform_add, 3, sizeof(cl_int), &group_offset);
    err |= clSetKernelArg(cl->kernel_uniform_add, 4, sizeof(cl_int), &base_index);
    err |= clSetKernelArg(cl->kernel_uniform_add, 5, sizeof(cl_int), &n);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "kernel_uniform_add: set kernel arguments");
    }

    err = CL_SUCCESS;
    err |= clEnqueueNDRangeKernel(cl->queue, cl->kernel_uniform_add, 1, NULL, global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        ERRX(EXIT_FAILURE, "kernel_uniform_add: execute kernel");
    }

    return CL_SUCCESS;
}

static int PreScanBufferRecursive(
    struct clconf *cl,
    struct databuf *db,
    cl_mem output_data, 
    cl_mem input_data, 
    int max_group_size,
    int max_work_item_count,
    int element_count, 
    int level)
{
    unsigned int group_size = max_group_size; 
    unsigned int group_count = (int)fmax(1.0f, (int)ceil((float)element_count / (2.0f * group_size)));
    unsigned int work_item_count = 0;

    if (group_count > 1)
        work_item_count = group_size;
    else if (IS_POWER_OF_TWO(element_count))
        work_item_count = element_count / 2;
    else
        work_item_count = floorPow2(element_count);
        
    work_item_count = (work_item_count > max_work_item_count) ? max_work_item_count : work_item_count;

    unsigned int element_count_per_group = work_item_count * 2;
    unsigned int last_group_element_count = element_count - (group_count-1) * element_count_per_group;
    unsigned int remaining_work_item_count = (int)fmax(1.0f, last_group_element_count / 2);
    remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
    unsigned int remainder = 0;
    size_t last_shared = 0;

    
    if (last_group_element_count != element_count_per_group)
    {
        remainder = 1;

        if(!IS_POWER_OF_TWO(last_group_element_count))
            remaining_work_item_count = floorPow2(last_group_element_count);    
        
        remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
        unsigned int padding = (2 * remaining_work_item_count) / NUM_BANKS;
        last_shared = sizeof(float) * (2 * remaining_work_item_count + padding);
    }

    remaining_work_item_count = (remaining_work_item_count > max_work_item_count) ? max_work_item_count : remaining_work_item_count;
    size_t global[] = { (int)fmax(1, group_count - remainder) * work_item_count, 1 };
    size_t local[]  = { work_item_count, 1 };  

    unsigned int padding = element_count_per_group / NUM_BANKS;
    size_t shared = sizeof(float) * (element_count_per_group + padding);
    
    cl_mem partial_sums = db->ScanPartialSums[level];
    int err = CL_SUCCESS;
    
    if (group_count > 1) {
        err = PreScanStoreSum(cl, global, local, shared, output_data, input_data, partial_sums, work_item_count * 2, 0, 0);
        if(err != CL_SUCCESS)
            return err;
            
        if (remainder) {
            size_t last_global[] = { 1 * remaining_work_item_count, 1 };
            size_t last_local[]  = { remaining_work_item_count, 1 };  

            err = PreScanStoreSumNonPowerOfTwo(cl,
                    last_global, last_local, last_shared, 
                    output_data, input_data, partial_sums,
                    last_group_element_count, 
                    group_count - 1, 
                    element_count - last_group_element_count);    
        
            if(err != CL_SUCCESS)
                return err;			
			
        }

        err = PreScanBufferRecursive(cl, db, partial_sums, partial_sums, max_group_size, max_work_item_count, group_count, level + 1);
        if(err != CL_SUCCESS)
            return err;
            
        err = UniformAdd(cl, global, local, output_data, partial_sums,  element_count - last_group_element_count, 0, 0);
        if(err != CL_SUCCESS)
            return err;
        
        if (remainder) {
            size_t last_global[] = { 1 * remaining_work_item_count, 1 };
            size_t last_local[]  = { remaining_work_item_count, 1 };  

            err = UniformAdd(cl,
                    last_global, last_local, 
                    output_data, partial_sums,
                    last_group_element_count, 
                    group_count - 1, 
                    element_count - last_group_element_count);
                
            if(err != CL_SUCCESS)
                return err;
        }
    }
    else if (IS_POWER_OF_TWO(element_count)) {
        err = PreScan(cl, global, local, shared, output_data, input_data, work_item_count * 2, 0, 0);
        if(err != CL_SUCCESS)
            return err;
    }
    else {
        err = PreScanNonPowerOfTwo(cl, global, local, shared, output_data, input_data, element_count, 0, 0);
        if(err != CL_SUCCESS)
            return err;
    }

    return CL_SUCCESS;
}

