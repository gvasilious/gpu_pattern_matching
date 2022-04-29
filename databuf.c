#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/param.h>
#include <CL/opencl.h> 

#include "databuf.h"
#include "common.h"
#include "ocl_context.h"
#include "utils.h"
#include "ocl_prefix_sum.h"
#include "ocl_compact_array.h"

#define GROUP_SIZE	256 /* XXX this is for the partial sum buffers only */

//#define COMPACT_RESULTS

int 
CreatePartialSumBuffers(struct databuf *db, cl_context ctx, unsigned int count) {
    unsigned int group_size = GROUP_SIZE; /* XXX pass it as argument */
    unsigned int element_count = count;

    int level = 0;

    do {       
        unsigned int group_count = (int)fmax(1, (int)ceil((float)element_count / (2.0f * group_size)));
        if (group_count > 1)
        {
            level++;
        }
        element_count = group_count;
        
    } while (element_count > 1);

    db->ScanPartialSums = (cl_mem*) malloc(level * sizeof(cl_mem));
    db->ScanPartialSums_size = level;
    memset(db->ScanPartialSums, 0, sizeof(cl_mem) * level);
    
    element_count = count;
    level = 0;
    
    do {       
        unsigned int group_count = (int)fmax(1, (int)ceil((float)element_count / (2.0f * group_size)));
        if (group_count > 1) {
            size_t buffer_size = group_count * sizeof(float);
            db->ScanPartialSums[level++] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        }

        element_count = group_count;

    } while (element_count > 1);

    return CL_SUCCESS;
}

void 
ReleasePartialSums(struct databuf *db) {
    unsigned int i;
    for (i = 0; i < db->ScanPartialSums_size; i++)
    {
        clReleaseMemObject(db->ScanPartialSums[i]);
    }    
    
    free(db->ScanPartialSums);

    db->ScanPartialSums = 0;
    db->ScanPartialSums_size = 0;
}



/*
 * creates a new data buffer
 * returns a pointer to the data buffer
 */
struct databuf *
databuf_new(size_t max_chunks, size_t max_chunk_size, int max_results,
    int mapped, struct clconf *clconf) //TODO XXX clconf should placed first arg
{
	int i;
	int e;
	struct databuf *db;

	cl_context ctx = clconf->ctx;
	cl_command_queue queue = clconf->queue;

	db = NULL;
	db = MALLOC(sizeof(struct databuf));
	if (!db)
		ERR(1, "ERROR: malloc db");

	db->cl                 = clconf;
	db->mapped             = mapped;
	db->max_results        = max_results;
	db->last_state         = 0;
	db->max_chunks         = max_chunks;
	db->max_chunk_size     = max_chunk_size;
	db->size               = db->max_chunks * db->max_chunk_size;
	db->results_comp_size  = db->max_chunks * db->max_chunk_size + 2; // one extra slot for total size; one for last state
	db->results2_comp_size = db->max_chunks * db->max_chunk_size + 2; // same as above
	db->chunks             = 0;
	db->bytes              = 0;


	/* device buffers */
	db->d_data = clCreateBuffer(ctx,
	    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	    db->size * sizeof(cl_uchar), NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_data: %s", clstrerror(e));

	db->d_indices = clCreateBuffer(ctx,
	    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	    db->max_chunks * sizeof(cl_int), NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_indices: %s", clstrerror(e));

	db->d_sizes = clCreateBuffer(ctx,
	    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	    db->max_chunks * sizeof(cl_int), NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_sizes: %s", clstrerror(e));

	db->d_results = clCreateBuffer(ctx,
	    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
	    (db->max_results * db->max_chunks + 1) * sizeof(cl_int), NULL, &e);

	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_results: %s", clstrerror(e));

	db->d_results2 = clCreateBuffer(ctx,
	    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
	    (db->max_results * db->max_chunks + 1) * sizeof(cl_int), NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_results2: %s", clstrerror(e));

	db->d_prefixsum = clCreateBuffer(ctx,
	    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	    db->max_chunks * sizeof(cl_int), NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_prefixsum: %s", clstrerror(e));

	db->d_results_comp = clCreateBuffer(ctx,
	    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
	    (db->results_comp_size) * sizeof(cl_int), NULL, &e);

	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_results_comp: %s", clstrerror(e));

	db->d_results2_comp = clCreateBuffer(ctx,
	    CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
	    (db->results2_comp_size) * sizeof(cl_int), NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_results2_comp: %s", clstrerror(e));

	CreatePartialSumBuffers(db, ctx, max_chunks); /* XXX:allocate pinned and mapped pointers */

	if (mapped) {
		/* mapped host buffers */
		db->h_data = clEnqueueMapBuffer(queue, db->d_data,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    db->size * sizeof(unsigned char), 0, NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_data: %s", clstrerror(e));

		db->h_indices = clEnqueueMapBuffer(queue, db->d_indices,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, 
		    db->max_chunks * sizeof(int), 0, NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_indices: %s", clstrerror(e));

		db->h_sizes = clEnqueueMapBuffer(queue, db->d_sizes,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    db->max_chunks * sizeof(int), 0, NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_sizes: %s", clstrerror(e));

		db->h_results = clEnqueueMapBuffer(queue, db->d_results,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    (db->max_results * db->max_chunks + 1) * sizeof(int), 0,
		    NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_results: %s", clstrerror(e));

		db->h_results2 = clEnqueueMapBuffer(queue, db->d_results2,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    (db->max_results * db->max_chunks + 1) * sizeof(int), 0,
		    NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_results2: %s", clstrerror(e));

		db->h_prefixsum = clEnqueueMapBuffer(queue, db->d_prefixsum,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    db->max_chunks * sizeof(int), 0, NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_prefixsum: %s", clstrerror(e));

		db->h_results_comp = clEnqueueMapBuffer(queue, db->d_results_comp,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    (db->results_comp_size) * sizeof(int), 0,
		    NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_results_comp: %s", clstrerror(e));

		db->h_results2_comp = clEnqueueMapBuffer(queue, db->d_results2_comp,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    (db->results2_comp_size) * sizeof(int), 0,
		    NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_results2_comp: %s", clstrerror(e));

	} else {
		/* unmapped host buffers */
		db->h_data = MALLOC(db->size * sizeof(unsigned char));
		if (!db->h_data)
			ERR(1, "ERROR: malloc h_data");

		db->h_indices = MALLOC(db->max_chunks * sizeof(int));
		if(!db->h_indices)
			ERR(1, "ERROR: malloc h_indices");

		db->h_sizes = MALLOC(db->max_chunks * sizeof(int));
		if(!db->h_sizes)
			ERR(1, "ERROR: malloc h_sizes");

		db->h_results = MALLOC((db->max_results * db->max_chunks + 1) *
		    sizeof(int));
		if(!db->h_results)
			ERR(1, "ERROR: malloc h_results");

		db->h_results2 = MALLOC((db->max_results * db->max_chunks + 1) *
		    sizeof(int));
		if(!db->h_results2)
			ERR(1, "ERROR: malloc h_results2");

		db->h_prefixsum = MALLOC(db->max_chunks * sizeof(int));
		if(!db->h_prefixsum)
			ERR(1, "ERROR: malloc h_prefixsum");

		db->h_results_comp = MALLOC((db->results_comp_size) *
		    sizeof(int));
		if(!db->h_results_comp)
			ERR(1, "ERROR: malloc h_results_comp");

		db->h_results2_comp = MALLOC((db->results2_comp_size) *
		    sizeof(int));
		if(!db->h_results2_comp)
			ERR(1, "ERROR: malloc h_results2_comp");

		/* pin host buffers */
		db->p_data = clCreateBuffer(ctx,
		    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		    db->size * sizeof(cl_uchar),db->h_data, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_data: %s", clstrerror(e));

		db->p_indices = clCreateBuffer(ctx,
		    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		    db->max_chunks * sizeof(cl_int), db->h_indices, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_indices: %s", clstrerror(e));

		db->p_sizes = clCreateBuffer(ctx,
		    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		    db->max_chunks * sizeof(cl_int), db->h_sizes, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_sizes: %s", clstrerror(e));

		db->p_results = clCreateBuffer(ctx,
		    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		    (db->max_results * db->max_chunks + 1) * sizeof(cl_int),
		    db->h_results, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_results: %s", clstrerror(e));

		db->p_results2 = clCreateBuffer(ctx,
		    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		    (db->max_results * db->max_chunks + 1) * sizeof(cl_int),
		    db->h_results2, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_results2: %s", clstrerror(e));

		db->p_prefixsum = clCreateBuffer(ctx,
		    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		    db->max_chunks * sizeof(cl_int), db->h_prefixsum, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_sizes: %s", clstrerror(e));

		db->p_results_comp = clCreateBuffer(ctx,
		    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		    (db->results_comp_size) * sizeof(cl_int),
		    db->h_results_comp, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_results_comp: %s", clstrerror(e));

		db->p_results2_comp = clCreateBuffer(ctx,
		    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
		    (db->max_chunk_size * db->max_chunks) * sizeof(cl_int),
		    db->h_results2_comp, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_results2_comp: %s", clstrerror(e));


	}

	/* per-chunk file ids */
	db->file_ids = NULL;
	db->file_ids = MALLOC(db->max_chunks * sizeof(int));
	if (!db->file_ids)
		ERR(1, "ERROR: malloc file_ids");

	/* initialize the meta-data */
	for (i = 0; i < max_chunks; i++) {
		db->h_sizes[i]   = db->max_chunk_size;
		db->h_indices[i] = db->max_chunk_size * i;
		db->file_ids[i]  = -1;
	}

	return db;
}


/*
 * adds bytes to the data buffer using file descriptor
 */
int
databuf_add_fd(struct databuf *db, int fd, int id, size_t *rd_bytes)
{
	int i;
	size_t size;
	size_t cur_chunks;


	size = read(fd, &db->h_data[db->h_indices[db->chunks]],
	    (db->max_chunks - db->chunks ) * db->max_chunk_size);

	*rd_bytes = size;

	/* this should never happen */
	if ((db->bytes + size) > db->size) {
		printf("ERROR: more data in buffer than maximum!\n");
		exit(EXIT_FAILURE);
	}

	/* the file descriptor had no more bytes */
	if (size == 0)
		return size;

	/* set the sizes and file ids of the new data */
	cur_chunks = size / db->max_chunk_size;
	for (i = db->chunks; i < db->chunks + cur_chunks; i++) {
		db->h_sizes[i]   = db->max_chunk_size;
		db->file_ids[i]  = id;
	}

	/* increase the chunks in the data buffer */
	db->chunks += cur_chunks;

	/* fix the data and metadata if the last chunk is not of maximum size */
	if (size % db->max_chunk_size != 0) {
		/* fix the size of the last chunk */
		db->h_sizes[db->chunks] = size - 
		    (cur_chunks * db->max_chunk_size);

		/* pad the rest of the chunk */
		for (i = 0; i < db->max_chunk_size - db->h_sizes[db->chunks];
		    i++) {
			db->h_data[db->h_indices[db->chunks] + 
			     db->h_sizes[db->chunks] + i] = 0;
		}

		/* assign it its file id */
		db->file_ids[db->chunks] = id;

		/* one more chunk for the missaligned one */
		db->chunks++;
	}

	/* current bytes in buffer */
	db->bytes = db->chunks * db->max_chunk_size;
	
	/* the buffer can not hold any more chunks */
	if (db->chunks == db->max_chunks) {
		return -1;
	}

	/* the buffer can not hold any more bytes */
	if (size == db->size) {
		printf("MAX_SIZE\n");
		return -2;
	}

	/* something went wrong with the chunks */
	if (db->chunks > db->max_chunks) {
		printf("ERROR: more chunks than maximum!\n");
		exit(EXIT_FAILURE);
	}

	/* something went wrong with the bytes */
	if (size > db->size) {
		printf("ERROR: more bytes than maximum\n");
		exit(EXIT_FAILURE);
	}

	/* the buffer can hold more data */
	return size;
}

/*
 * adds lines to the data buffer using file descriptor
 */
int
databuf_add_fp(struct databuf *db, FILE *fp, int id, int aligned, size_t *rd_bytes, size_t *rd_lines)
{
	char *buf;
	unsigned int toread;
	unsigned int len;
	int i;

	*rd_bytes = *rd_lines = 0;

	buf = &db->h_data[db->bytes]; /* next slot to write */
	toread = MIN(db->size - db->bytes, db->max_chunk_size); /* bytes to
								  read */
	while (fgets(buf, toread, fp) != NULL) {

		len = strnlen(buf, toread);

		*rd_bytes += len;

		if (buf[len - 1] == '\n') {
			*rd_lines += 1;
		} else {
			/* line was truncated; this happens when a line
			 * is larger than db->max_chunk_size or when the
			 * available space is not enough */ ; 
		}

		/* set the indices, sizes and file ids of the new data */
		db->h_indices[db->chunks] = db->bytes;
		db->h_sizes[db->chunks] = len;
		db->file_ids[db->chunks] = id;

		/* increase the chunks in the data buffer */
		db->chunks += 1;

		/* increase bytes in the data buffer */
		if (aligned) {
			db->bytes += ROUNDUP(len, 16);
			if (db->bytes > db->size)
				db->bytes = db->size;
		} else {
			db->bytes += len;
			if (db->bytes > db->size)
				db->bytes = db->size;
		}

		/* zero padding space; it is needed to prevent matching
		 * previously-stored data */
		if (aligned) {
			memset(&buf[len], 0, ROUNDUP(len, 16) - len);
		}

		/* the buffer cannot hold any more chunks */
		if (db->chunks >= db->max_chunks) {
			return -1;
		}

		/* the buffer cannot hold any more data */
		if (db->bytes >= db->size) {
			return -2;
		}

		buf = &db->h_data[db->bytes]; /* next slot to write */
		toread = MIN(db->size - db->bytes, db->max_chunk_size);
	}

	/* the buffer can hold more data */
	return db->size - db->bytes;

}


/*
 * adds a single chunk of len bytes to the data buffer
 */
int
databuf_add_chunk(struct databuf *db, char *chunk, size_t len, int id, char aligned)
{
	int i;

	/* chunk is too large */
	if (len > db->max_chunk_size) {
		return -3;
	}

	/* the buffer cannot hold any more chunks */
	if (db->chunks >= db->max_chunks) {
		return -1;
	}

	/* the buffer cannot hold this chunk */
	if (db->bytes + len >= db->size) {
		return -2;
	}

	memcpy(&db->h_data[db->bytes], chunk, len);

	/* set the indices, sizes and file ids of the new data */
	db->h_indices[db->chunks] = db->bytes;
	db->h_sizes[db->chunks] = len;
	db->file_ids[db->chunks] = id;

	/* increase the chunks in the data buffer */
	db->chunks += 1;

	/* current bytes in buffer */
	if (aligned) {
		db->bytes += ROUNDUP(len, 16);
		if (db->bytes > db->size)
			db->bytes = db->size;
	} else {
		db->bytes += len;
	}

	/* the buffer can hold more data */
	return db->size - db->bytes;
}

/*
 * resets the data buffer for reuse
 */
void
databuf_reset(struct databuf *db)
{
	db->chunks = 0;
	db->bytes  = 0;

	return;
}


/*
 * clears the data buffer
 */
void
databuf_clear(struct databuf *db)
{
	memset(db->h_data, 0,
			db->size);
	memset(db->h_indices, 0,
			db->max_chunks * sizeof(int));
	memset(db->h_sizes, 0,
			db->max_chunks * sizeof(int));
	memset(db->h_results, 0,
			(db->max_chunks * db->max_results + 1) * sizeof(int));
	memset(db->h_results2, 0,
			(db->max_chunks * db->max_results + 1) * sizeof(int));
	memset(db->h_results_comp, 0,
			(db->results_comp_size) * sizeof(int));
	memset(db->h_results2_comp, 0,
			(db->results2_comp_size) * sizeof(int));
	memset(db->file_ids, 0,
			db->max_chunks * sizeof(int));

	databuf_reset(db);

	return;
}

/*
 * copies the data buffer to the device
 */
void
databuf_copy_host_to_device(struct databuf *db, cl_command_queue queue)
{
	int e;

	/* if the buffer is mapped, there is nothing to do */
	if (db->mapped)
		return;

	e = clEnqueueWriteBuffer(queue, db->d_data, CL_TRUE, 0,
	    db->bytes * sizeof(cl_uchar), db->h_data, 0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: write d_data: %s", clstrerror(e));
	e = clEnqueueWriteBuffer(queue, db->d_indices, CL_TRUE, 0,
	    db->chunks * sizeof(cl_int), db->h_indices, 0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: write d_indices: %s", clstrerror(e));
	e = clEnqueueWriteBuffer(queue, db->d_sizes, CL_TRUE, 0,
	    db->chunks * sizeof(cl_int), db->h_sizes, 0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: write d_sizes: %s", clstrerror(e));

	return;
}


/*
 * copies the data buffer to the device
 */
void
databuf_copy_device_to_host(struct databuf *db, cl_command_queue queue)
{
	int e;

	/* if the buffer is mapped, there is nothing to do */
	if (db->mapped) {
		db->last_state = db->h_results[db->chunks * db->max_results];
		return;
	}

	/* XXX TODO do not copy d_results array if COMPACT_RESULTS is enabled */
	e = clEnqueueReadBuffer(queue, db->d_results, CL_TRUE, 0,
	    (db->max_results * db->chunks + 1) * sizeof(cl_int), db->h_results,
	    0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: read d_results: %s", clstrerror(e));

	// XXX TODO Take care of last_state when COMPACT_RESULTS is enabled
	db->last_state = db->h_results[db->chunks * db->max_results];

//#define COMPACT_RESULTS

#ifndef COMPACT_RESULTS
	e = clEnqueueReadBuffer(queue, db->d_results2, CL_TRUE, 0,
	    (db->max_results * db->chunks + 1) * sizeof(cl_int), db->h_results2,
	    0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: read d_results2: %s", clstrerror(e));

#else // COMPACT_RESULTS

	///* no need to copy the whole prefixsum array */
	//e = clEnqueueReadBuffer(queue, db->d_prefixsum, CL_TRUE, 0,
	//    (db->chunks) * sizeof(cl_int), db->h_prefixsum,
	//    0, NULL, NULL);
	//if (e != CL_SUCCESS)
	//	ERRXV(1, "ERROR: read d_prefixsum: %s", clstrerror(e));
	//
	///* compute total number of matches */
	//int matches_total = db->h_prefixsum[db->chunks - 1] + db->h_results[db->chunks - 1];
	//printf("[prefixsum] Total number of matches: %d\n", matches_total);


	/* compute prefix sums; will be used to do array compaction */
	ocl_prefix_sum(db->cl, db, db->chunks);

	/* do the array compaction */
	ocl_compact_array(db->cl, db, 1024 /*ctx->local_ws*/); //TODO: XXX find a way to pass local_ws as a parameter


	/* get first two elements of the array: the last state and
	 * the total matches */
	e = clEnqueueReadBuffer(queue, db->d_results_comp, CL_TRUE, 0,
	    (2) * sizeof(cl_int), db->h_results_comp,
	    0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: read d_results_comp: %s", clstrerror(e));


	/* -1 due to the fact that the first element is used for keeping
	 * the length of the array.
	 * XXX should be -2 for the length and the last state. */
	int m = min(db->h_results_comp[0], db->results_comp_size - 1);
	printf("[results_comp[0]] Total number of matches: %d\n", m);

	if (m > 0) {
		e = clEnqueueReadBuffer(queue, db->d_results_comp, CL_TRUE, 0,
		    		(m + 2) * sizeof(cl_int), db->h_results_comp,
		    		0, NULL, NULL);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: read d_results_comp: %s", clstrerror(e));

		e = clEnqueueReadBuffer(queue, db->d_results2_comp, CL_TRUE, 0,
		    		(m + 2) * sizeof(cl_int), db->h_results2_comp,
		    		0, NULL, NULL);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: read d_results2_comp: %s", clstrerror(e));

		db->last_state = db->h_results_comp[m+1];
	
		//for(int i=0;i<db->chunks;i++) {
		//	printf("%d ", db->h_results[i]);
		//}
		//printf("\n");
		//for(int i=0;i<db->chunks;i++) {
		//	printf("%d ", db->h_prefixsum[i]);
		//}
		//printf("\n");
		//for(int i=0;i<db->chunks;i++) {
		//	printf("%d ", db->h_prefixsum[db->chunks+i]);
		//}
		//printf("\n");
		//
		//for(int i=0;i<m+1;i++) {
		//	printf("%d ", db->h_results_comp[i]);
		//}
		//printf("\n");
		//for(int i=0;i<m+1;i++) {
		//	printf("%d ", db->h_results2_comp[i]);
		//}
		//printf("\n");
	}
#endif
	return;
}

/*
 * Execute callback function on the results.
 */
int
databuf_process_results_compact(struct databuf *db, int (*cb)(int file_idx, int patrn_idx, int chunk_idx, int offset, void* uarg), void *uarg)
{
	int i, j, matches = 0, max_results = 0;
	int pat_index, pat_id, file_id, pat_len = 0, offset = 0;
	unsigned char *pat_name;
	char *fname;
	int *res, *res2;
	int c_idx = 0;

	res = db->h_results_comp;
	res2 = db->h_results2_comp;
	max_results = db->max_chunks * db->max_chunk_size;
	matches = db->h_results_comp[0];

	/* loop the results array for every chunk of this databuf */
	for (i = 0; i < matches && i < db->results_comp_size; i++) {
		pat_index = res[i + 1];
		offset    = res2[i + 1]; /* XXX find pat_len and substract */
		c_idx     = offset / db->max_chunk_size;
		file_id   = db->file_ids[c_idx]; //XXX do some math here to compute the file id

		if (cb) {
			cb(file_id, pat_index, c_idx, offset, uarg);
		}
	}

	/* return total matches */
	return matches;
}

/*
 * Execute callback function on the results.
 */
int
databuf_process_results_buckets(struct databuf *db, int (*cb)(int file_idx, int patrn_idx, int chunk_idx, int offset, void* uarg), void *uarg)
{
	int i, j, matches = 0, max_results = 0;
	int pat_index, pat_id, file_id, pat_len = 0, offset = 0;
	unsigned char *pat_name;
	char *fname;
	int *res, *res2;

	res = db->h_results;
	res2 = db->h_results2;
	max_results = db->max_results;

	/* loop the results array for every chunk of this databuf */
	for (i = 0; i < db->chunks; i++) {
		matches += db->h_results[i];

		/* print the patterns found if verbose is on */
		if (res[i] > 0) {
			for (j = 0;
			    j < res[i] && (j < max_results - 1);
			    j++) {
				pat_index = res[(j+1)*db->chunks + i];
				/* XXX pat_len has never instantiated; */
				offset    = res2[(j+1)*db->chunks + i] - pat_len + 1; /* XXX why need to +1 in the offset? */
				file_id = db->file_ids[i];

				if (cb)
					cb(file_id, pat_index, i, offset, uarg);
			}
		}
	}

	/* return total matches */
	return matches;
}

/*
 * Execute callback function on the results.
 */
int
databuf_process_results(struct databuf *db, int (*cb)(int file_idx, int patrn_idx, int chunk_idx, int offset, void* uarg), void *uarg) {
#ifdef COMPACT_RESULTS
	databuf_process_results_compact(db, cb, uarg);
#else
	databuf_process_results_buckets(db, cb, uarg);
#endif
}


/*
 * frees the data buffer 
 */
void
databuf_free(struct databuf *db, int mapped, cl_command_queue queue)
{
	if (mapped) {
		clEnqueueUnmapMemObject(queue, db->d_data, db->h_data,
		    0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, db->d_indices, db->h_indices,
		    0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, db->d_sizes, db->h_sizes,
		    0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, db->d_results, db->h_results,
		    0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, db->d_results2, db->h_results2,
		    0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, db->p_prefixsum, db->h_prefixsum,
		    0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, db->d_results_comp, db->h_results_comp,
		    0, NULL, NULL);
		clEnqueueUnmapMemObject(queue, db->d_results2_comp, db->h_results2_comp,
		    0, NULL, NULL);

	} else {
		FREE(db->h_data);
		FREE(db->h_indices);
		FREE(db->h_sizes);
		FREE(db->h_results);
		FREE(db->h_results2);
		FREE(db->h_prefixsum);
		FREE(db->h_results_comp);
		FREE(db->h_results2_comp);
		FREE(db->file_ids);
	}

	clReleaseMemObject(db->d_data);
	clReleaseMemObject(db->d_indices);
	clReleaseMemObject(db->d_sizes);
	clReleaseMemObject(db->d_results);
	clReleaseMemObject(db->d_results2);
	clReleaseMemObject(db->d_prefixsum);
	clReleaseMemObject(db->d_results_comp);
	clReleaseMemObject(db->d_results2_comp);

	ReleasePartialSums(db);

	FREE(db);

	return;
}

#ifdef DATABUF_TEST

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ocl_prefix_sum.h"
#include "ocl_compact_array.h"

int main(int argc, char *argv[]) {

	struct databuf *b;
	struct clconf cl;
	FILE *fp;
	char buf[128];
	char *line = NULL;
	unsigned char *chunk;
	ssize_t len = 0;
	int dev_pos = 0;
	unsigned int i, j;
	size_t databuf_sz = 1024;

	if (argc != 2) {
		printf("usage: %s <size>\n", argv[0]);
		return 0;
	}

	databuf_sz = atoi(argv[1]);

	srand(time(NULL));

	clinitctx(&cl, 0, -1);

	ocl_prefix_sum_init(&cl);

	ocl_compact_array_init(&cl);


	/********************************************************************/

	printf("Testing databuf creation... ");

	b = databuf_new(/*max_chunks*/ databuf_sz, /* max_chunk_size */ 80,
			/*max_results */128 + 1, /*mapped*/1, &cl);

	if (b == NULL) {
		printf("FAILED\n");
		goto end;
	} else {
		printf("OK\n");
	}

	/********************************************************************/

	printf("Testing insert operations... ");
	for (i=0; i<databuf_sz; i++) {
		sprintf(buf, "test%d", i);
		if (databuf_add_chunk(b, buf, strlen(buf), i, 1) < 0) {
			printf("FAILED\n");
			goto end;
		}
	}

	/* Trying to insert one more */
	if (databuf_add_chunk(b, buf, strlen(buf), i, 1) > 0) {
		printf("FAILED\n");
		goto end;
	}

	/* Read and validate chunks from databuf */
	for (i=0; i < b->chunks; i++) {
		chunk = &b->h_data[b->h_indices[i]];
		sprintf(buf, "test%d", i);

		for (j=0; j < b->h_sizes[i]; j++) {
			if (buf[j] != chunk[j]) {
				printf("FAILED\n");
				goto end;
			}
		}
	}
	printf("OK\n");

	/********************************************************************/

	printf("Testing prefix sums... ");

	/* fill results array with random data */
	int count = 0; // this is the total number of matches
	for (i=0; i < b->max_chunks; i++) {
		int r = rand() % (b->max_results);
		b->h_results[i] = r;
		b->h_results2[i] = r;

		for (j=0; j < r; j++) {
			b->h_results [(j+1) * b->chunks + i] = count; //XXX fill up with pattern idx
			b->h_results2[(j+1) * b->chunks + i] = count; //XXX fill up with offest
			count += 1;
		}
	}

	int e = clEnqueueWriteBuffer(cl.queue, b->d_results, CL_TRUE, 0,
	    (b->max_results * b->max_chunks + 1) * sizeof(int), b->h_results, 0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: write d_results: %s", clstrerror(e));

	e = clEnqueueWriteBuffer(cl.queue, b->d_results2, CL_TRUE, 0,
	    (b->max_results * b->max_chunks + 1) * sizeof(int), b->h_results2, 0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: write d_results2: %s", clstrerror(e));

	/* compute prefix sums; will be used to do array compaction */
	ocl_prefix_sum(&cl, b, b->chunks);

	e = clEnqueueReadBuffer(cl.queue, b->d_prefixsum, CL_TRUE, 0,
	    (b->max_chunks) * sizeof(int), b->h_prefixsum,
	    0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: read d_prefixsum: %s", clstrerror(e));

	/* compute total number of matches */
	int matches_total = b->h_prefixsum[b->chunks - 1] + b->h_results[b->chunks - 1];

	count = 0;
	for (i=0; i<b->max_chunks; i++) {
		//printf("%d ", b->h_prefixsum[i]);
		if (b->h_prefixsum[i] != count) {
			printf("FAILED\n");
			goto end;
		}
		count += b->h_results[i];
	}

	printf("OK\n");

	/********************************************************************/

	printf("Testing compact array... ");

	ocl_compact_array(&cl, b, 256);

	e = clEnqueueReadBuffer(cl.queue, b->d_results_comp, CL_TRUE, 0,
	    (matches_total + 1) * sizeof(cl_int), b->h_results_comp,
	    0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: read d_results_comp: %s", clstrerror(e));

	e = clEnqueueReadBuffer(cl.queue, b->d_results2_comp, CL_TRUE, 0,
	    (matches_total + 1) * sizeof(cl_int), b->h_results2_comp,
	    0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: read d_results2_comp: %s", clstrerror(e));

	// check that total matches are corectly reported
	if (matches_total != b->h_results_comp[0]
			|| matches_total != b->h_results2_comp[0]) {
		printf("FAILED\n");
		goto end;
	}

	// check reported matches one by one
	for (i=0; i < matches_total; i++) {
		if (i != b->h_results_comp[i+1]
				&& i != b->h_results2_comp[i+1]) {
			printf("FAILED\n");
			goto end;
		}
		//printf("%d ", b->h_results_comp[i+1]);
		//printf("%d ", b->h_results2_comp[i+1]);
	}

	printf("OK\n");
	/********************************************************************/

	printf("Resetting databuf... ");
	databuf_reset(b);
	printf("DONE\n");

	printf("Adding /etc/motd lines into databuf... ");
    	fp = fopen("/etc/motd", "r");
    	if (fp == NULL)
		exit(EXIT_FAILURE);

	count = 0;
    	while ((fgets((char*)&buf, sizeof(buf), fp) != NULL) && (count < databuf_sz)) {
		len = strlen(buf);

		if (len > 1 && (buf[len - 1] == '\n'))
			buf[len - 1] = '\0';

		if (databuf_add_chunk(b, buf, len, 0, 1) < 0) {
			printf("TEST_FAILED\n");
			goto end;
		}
		count++;
    	}

    	fclose(fp);
    	if (line)
	    	free(line);
	printf("DONE\n");

	printf("Resetting databuf... ");
	databuf_reset(b);
	printf("DONE\n");

	printf("Adding /etc/motd lines into databuf... ");
    	fp = fopen("/etc/motd", "r");
    	if (fp == NULL)
		exit(EXIT_FAILURE);

	size_t bytes_total = 0;
	size_t lines_total = 0;
	do {
		e = databuf_add_fp(b, fp, 0, 1, &bytes_total, &lines_total);
	} while (e != -1 && e != -2 && !feof(fp));

    	fclose(fp);
	printf("DONE\n");

#if 0
	for (i=0; i < b->chunks; i++) {
		chunk = &b->h_data[b->h_indices[i]];
		printf("[ind:%d][len:%d] ", b->h_indices[i], b->h_sizes[i]);
		for (j=0; j < b->h_sizes[i]; j++) {
		       printf("%c", chunk[j]);
		}
		//printf("\n");
	}
#endif

end:
	return 0;
}
#endif

