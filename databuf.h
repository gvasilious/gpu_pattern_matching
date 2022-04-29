#ifndef _DATABUF_H_
#define _DATABUF_H_

#include <CL/opencl.h>

#include "ocl_context.h"

/* maximum number of result cells per chunk */
#define MAX_RESULTS 16


/*
 * data buffer
 */
struct databuf {
	unsigned char	*h_data;	 /* host data array                 */
	int 		*h_indices;	 /* host chunk indices array        */
	int		*h_sizes;	 /* host chunk sizes array          */
	int		*h_results;	 /* host results array (pattern id) */
	int		*h_results2;	 /* host results array (offset)     */
	int		*h_prefixsum;	 /* host prefix sums array          */
	int		*h_results_comp; /* host compacted results array;
					  * first element is the number of
					  * results this array has.         */
	int		*h_results2_comp;/* host compacted results2 array;
     					  * first element is the number of
					  * results this array has.         */

	size_t		results_comp_size; /* the size of h_results_comp    */
	size_t		results2_comp_size;/* the size of h_results2_comp   */

	int		*file_ids;	 /* file ID per chunk               */ 
	int		mapped;		 /* memory mapped buffer flag       */
	int		max_results;	 /* maximum result cells per chunk  */
	long		last_state;	 /* last AC state at the last chunk */
	size_t		max_chunks;	 /* maximum number of chunks        */
	size_t		max_chunk_size;	 /* maximum chunk size (Bytes)      */
	size_t		size;		 /* data buffer size (Bytes)        */
	size_t		chunks;		 /* number of chunks in this buffer */
	size_t		bytes;		 /* current data bytes in buffer    */

	cl_mem		d_data;		 /* device data array               */
	cl_mem		d_indices;	 /* device chunk indices array      */
	cl_mem		d_sizes;	 /* device chunk sizes array        */
	cl_mem		d_results;	 /* device results array            */
	cl_mem		d_results2;	 /* device results array            */
	cl_mem		d_prefixsum;     /* device prefix sums array        */
	cl_mem		d_results_comp;  /* device results compacted array  */
	cl_mem		d_results2_comp; /* device results2 compacted array */

	cl_mem		p_data;		 /* pinned memory for data          */
	cl_mem		p_indices;	 /* pinned memory for indices       */
	cl_mem		p_sizes;	 /* pinned memory for sizes         */
	cl_mem		p_results;	 /* pinned memory for results       */
	cl_mem		p_results2;	 /* pinned memory for results       */
	cl_mem		p_prefixsum;     /* pinned memory for prefix sums   */
	cl_mem		p_results_comp;  /* pinned memory for results       */
	cl_mem		p_results2_comp; /* pinned memory for results       */

	cl_mem		*ScanPartialSums;
	unsigned int	ScanPartialSums_size;

	struct clconf	*cl;
};


/*
 * creates a new data buffer
 * returns a pointer to the data buffer
 *
 * arg0: maximum number of chunks
 * arg1: maximum chunk size
 * arg2: maximum result cells per chunk
 * arg3: mapped buffer flag
 * arg4: OpenCL conf
 *
 * ret:  a new data buffer object
 */
struct databuf *
databuf_new(size_t, size_t, int, int, struct clconf*); 


/*
 * adds bytes to the data buffer using file descriptor
 *
 * arg0: data buffer
 * arg1: file descriptor
 * arg2: file id
 * arg3: read bytes counter
 *
 * ret:   1 if the buffer can hold more data after this call
 *       -1 if the buffer is full of chunks
 *       -2 if the buffer is full of bytes
 * ret:  always returns the read bytes via arg3
 */
int
databuf_add_fd(struct databuf *, int, int, size_t *);

/*
 * adds lines to the data buffer using file pointer
 *
 * arg0: data buffer
 * arg1: file pointer
 * arg2: file id
 * arg3: whether data will be stored aligned
 * arg4: read bytes counter
 * arg5: read lines counter
 *
 * ret:   1 if the buffer can hold more data after this call
 *       -1 if the buffer is full of chunks
 *       -2 if the buffer is full of bytes
 * ret:  always returns the read bytes and read lines via arg4 and arg5
 */
int
databuf_add_fp(struct databuf *, FILE *, int, int, size_t *, size_t*);


/*
 * resets the data buffer for reuse
 *
 * arg0: data buffer
 */
void
databuf_reset(struct databuf *);


/*
 * clears the data buffer
 *
 * arg0: data buffer
 */
void
databuf_clear(struct databuf *);


/*
 * copies the data buffer to the device
 *
 * arg0: data buffer
 * arg1: OpenCL command queue
 */
void
databuf_copy_host_to_device(struct databuf *, cl_command_queue);


/*
 * copies the data buffer to the device
 *
 * arg0: data buffer
 * arg1: OpenCL command queue
 */
void
databuf_copy_device_to_host(struct databuf *, cl_command_queue);


/*
 * Execute callback function on the results
 *
 * arg0: data buffer
 * arg1: callback function for each match found
 * arg2: user argument
 */
int
databuf_process_results(struct databuf *db, int (*cb)(int file_idx, int patrn_idx, int chunk_idx, int offset, void* uarg), void *uarg);


/*
 * frees the data buffer
 *
 * arg0: data buffer
 * arg2: OpenCL command queue
 */
void
databuf_free(struct databuf *, int, cl_command_queue);


#endif /* _DATABUF_H_ */
