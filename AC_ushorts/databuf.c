/*
 * authors:	Dimitris Deyannis
 * 		Eva Papadogiannaki
 * last update:	Mar-2017
 * 		Nov-2018
 * contact:	deyannis@ics.forth.gr
 * 		epapado@ics.forth.gr
 */

#include <stdio.h>
#include <stdlib.h>

#include "databuf.h"
#include "common.h"
#include "clutil.h"

#include <CL/opencl.h> 


/*
 * creates a new data buffer
 * returns a pointer to the data buffer
 */
struct databuf *
databuf_new(size_t max_chunks, size_t max_chunk_size, int max_results,
    int mapped, cl_context ctx, cl_command_queue queue)
{
	int i;
	int e;
	struct databuf *db;

	db = NULL;
	db = MALLOC(sizeof(struct databuf));
	if (!db)
		ERR(1, "ERROR: malloc db");

	db->mapped         = mapped;
	db->max_results    = max_results;
	db->last_state     = 0;
	db->max_chunks     = max_chunks;
	db->max_chunk_size = max_chunk_size;
	db->size           = db->max_chunks * db->max_chunk_size * sizeof(unsigned short);
	db->chunks         = 0;
	db->bytes          = 0;


	/* device buffers */
	db->d_data = clCreateBuffer(ctx,
	    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	    db->size, NULL, &e); 
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
	    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
	    (db->max_results * db->max_chunks + 1) * sizeof(cl_int), NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_results: %s", clstrerror(e));

	if (mapped) {
		/* mapped host buffers */
		db->h_data = clEnqueueMapBuffer(queue, db->d_data,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0,
		    db->size, 0, NULL, NULL, &e); 
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
	} else {
		/* unmapped host buffers */
		db->h_data = MALLOC(db->size); 
		if (!db->h_data)
			ERR(1, "ERROR: malloc h_data");
		db->h_indices = MALLOC(db->max_chunks * sizeof(int));
		if(!db->h_indices)
			ERR(1, "ERROR: malloc h_indices");
		db->h_sizes = MALLOC(db->max_chunks * sizeof(int));
		if(!db->h_indices)
			ERR(1, "ERROR: malloc h_sizes");
		db->h_results = MALLOC((db->max_results * db->max_chunks + 1) *
		    sizeof(int));
		if(!db->h_indices)
			ERR(1, "ERROR: malloc h_results");

		/* pin host buffers */
		db->p_data = clCreateBuffer(ctx,
		    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		    db->size, db->h_data, &e); 
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
		    CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		    (db->max_results * db->max_chunks + 1) * sizeof(cl_int),
		    db->h_results, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: pin h_results: %s", clstrerror(e));
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
databuf_add_fd(struct databuf *db, FILE *fp, int id, size_t *rd_bytes, size_t *wr_bytes)
{
	#define MAX_CHAR 16384
	int i, nt;
	size_t rd_size, wr_size;
	size_t cur_chunks;
	char *strdata, *token;
	char *signature_pattern;

	rd_size = 0;
	wr_size = 0;


	strdata = MALLOC(MAX_CHAR * sizeof(char));
	if(strdata == NULL)
		ERR(1, "ERROR: malloc strdata");

	strdata = fgets(strdata, MAX_CHAR * sizeof(char), fp);
	if (strdata == NULL) {
		rd_size = 0;
		*rd_bytes = rd_size;
		*wr_bytes = rd_size;
		return rd_size;
	}
	rd_size = strlen(strdata);
	strdata[strlen(strdata)-1] = '\0';
	signature_pattern = strtok(strdata, ";");

	nt = 0;
	token = strtok(signature_pattern, ",");
	while (token) {
		db->h_data[db->h_indices[db->chunks] + (nt * sizeof(unsigned short))] = (unsigned short) atoi(token);
		wr_size += sizeof(unsigned short);
		token = strtok(NULL, ",");
		nt++;
	}
	free(strdata);	
	*rd_bytes = rd_size;
	*wr_bytes = wr_size;

	/* this should never happen */
	if ((db->bytes + wr_size) > db->size) {
		printf("ERROR: more data in buffer than maximim!\n");
		exit(EXIT_FAILURE);
	}

	/* the file descriptor had no mode bytes */
	if (rd_size == 0)
		return rd_size;

	/* set the sizes and file ids of the new data */
	cur_chunks = wr_size / db->max_chunk_size;
	for (i = db->chunks; i < db->chunks + cur_chunks; i++) {
		db->h_sizes[i]   = db->max_chunk_size;
		db->file_ids[i]  = id;
	}

	/* increase the chunks in the data buffer */
	db->chunks += cur_chunks;
	/* fix the data and metadata if the last chunk is not of maximum size */
	if (wr_size % db->max_chunk_size != 0) {

		/* fix the size of the last chunk */
		db->h_sizes[db->chunks] = wr_size - 
		    (cur_chunks * db->max_chunk_size);

		/* pad the rest of the chunk */
		for (i = nt % (db->max_chunk_size/sizeof(unsigned short)); i < (db->max_chunk_size / sizeof(unsigned short)); i++) {
			db->h_data[db->h_indices[db->chunks] + (i * sizeof(unsigned short))] = 0;
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
	if (wr_size == db->size) {
		printf("MAX_SIZE\n");
		return -2;
	}

	/* something went wrong with the chunks */
	if (db->chunks > db->max_chunks) {
		exit(EXIT_FAILURE);
	}

	/* something went wrong with the bytes */
	if (wr_size > db->size) {
		printf("ERROR: more bytes than maximum\n");
		exit(EXIT_FAILURE);
	}

	/* the buffer can hold more data */
	return rd_size;
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
	memset(db->h_data,    0, db->size); 
	memset(db->h_indices, 0, db->max_chunks * sizeof(int));
	memset(db->h_sizes,   0, db->max_chunks * sizeof(int));
	memset(db->h_results, 0, db->max_chunks * sizeof(int));
	memset(db->file_ids,  0, db->max_chunks * sizeof(int));
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
	    db->bytes * sizeof(ushort), db->h_data, 0, NULL, NULL); 
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

	e = clEnqueueReadBuffer(queue, db->d_results, CL_TRUE, 0,
	    (db->max_results * db->chunks + 1) * sizeof(cl_int), db->h_results,
	    0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: read d_results: %s", clstrerror(e));

	db->last_state = db->h_results[db->chunks * db->max_results];

	return;
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
	} else {
		FREE(db->h_data);
		FREE(db->h_indices);
		FREE(db->h_sizes);
		FREE(db->h_results);
		FREE(db->file_ids);
	}

	clReleaseMemObject(db->d_data);
	clReleaseMemObject(db->d_indices);
	clReleaseMemObject(db->d_sizes);
	clReleaseMemObject(db->d_results);

	FREE(db);

	return;
}
