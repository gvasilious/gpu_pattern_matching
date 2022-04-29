__kernel void
ahomatch(__global int *trans, __global uint4 *data, __global int *indices, 
    __global int *sizes, __global int*results, __global int*results2,
    const unsigned int chunks, const unsigned long data_size,
    const long last_state, const int max_pat_size, const int max_results)
{
#define ALPHABET_SIZE	256
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

	int i;
	int j;
	int id;
	int index;
	int size;
	int matches = 0; // count the matches per thread
	long state, state_prev;
	unsigned char c;
	unsigned char *p_c16;
	uint4 c16;

	id = get_global_id(0);
	matches = 0;
	for (i = 0; i < max_results; i++) {
		results[max_results * id + i] = -1;
		//results2[max_results * id + i] = -1;
	}
	if (id >= chunks)
		goto end;

	index = indices[id];
	size = sizes[id];

	/*
	 * The first thread is assigned with the state of the last tread of 
	 * the previous kernel call so we can grep the matches splitted over 
	 * two data buffers
	 */
	if (id == 0)
		state = last_state;
	else
		state = 0;

	size = CEILDIV(size, sizeof(uint4));

	/* fetch 16 chars */
	for (i = 0; i < size; i++) {
		c16 = *(data + index / sizeof(uint4) + i);
		p_c16 = (unsigned char *)&c16;

		/* loop on the fetched data */
		for (j = 0; j < sizeof(uint4); j++) {
			c = p_c16[j];

			state_prev = state;
			state = *(trans + (unsigned long)(ALPHABET_SIZE * 2) *
			    (unsigned long)state + (unsigned long)c);

			/* match */
			if (state < 0) {
				matches++;
				state = -state;
				if (matches < max_results) {
					results[max_results * id + matches] =
					    *(trans +
					    (unsigned long)(ALPHABET_SIZE * 2) *
					    (unsigned long)(state_prev) +
					    (unsigned long)c +
					    (unsigned long)ALPHABET_SIZE);
					results2[max_results * id + matches] = i * sizeof(uint4) + j;
				}
			}
		}
	}


	/*
	 * The last thread saves its state for the first thread of the next
	 * bufferand returns. This will grep the matches splitted over two data
	 * buffers
	 */
	if (id == chunks -1) {
		results[chunks * max_results] = state;
		goto end;
	}


	/*
	 * All threads except for the last one reach this point. If their last 
	 * state is zero, they do not have a match in progress so they return
	 */
	if (state == 0)
		goto end;


	/* 
	 * All threads except for the final that have a match in progress after
	 * they have consumed their data chunk continue up to this step. We let 
	 * them continue over the next thread's data for max pattern size bytes.
	 * This will grep the matches splitted over the data chunks of two
	 * different threads
	 */
	size += (CEILDIV(max_pat_size, sizeof(uint4)));

	/* fetch 16 chars */
	for ( ; i < size; i++) {
		/* guard the end of data buffer */
		if (i * sizeof(uint4) + index + sizeof(uint4) > data_size)
			goto end;

		c16 = *(data + index / sizeof(uint4) + i);
		p_c16 = (unsigned char *)&c16;

		/* loop on the fetched data */
		for (j = 0; j < sizeof(uint4); j++) {
			c = p_c16[j];

			state_prev = state;
			state = *(trans + (unsigned long)(ALPHABET_SIZE * 2) *
			    (unsigned long)state + (unsigned long)c);

			/*
			 * If the continued match fails, return here so 
			 * this thread does not grep duplicate matches
			 */
			if (state == 0)
				goto end;

			/* match */
			if (state < 0) {
				matches++;
				state = -state;
				if (matches < max_results) {
					results[max_results * id + matches] =
					    *(trans +
					    (unsigned long)(ALPHABET_SIZE * 2) *
					    (unsigned long)(state_prev) +
					    (unsigned long)c +
					    (unsigned long)ALPHABET_SIZE);
					results2[max_results * id + matches] = i * sizeof(uint4) + j;
				}

				/* ATTENTION HERE
				 * This one might make you lose some matches 
				 * depending on your application
				 */
				goto end;
			}
		}
	}

end:
	results[max_results * id] = matches;
	results2[max_results * id] = matches;


	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////
#if 0
	uint n_items = get_global_size(0);
	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	uint dp = 1;

	__global int *a = results;
	__global int *b = results2;

	b[2*lid] = a[2*gid];
	b[2*lid+1] = a[2*gid+1];

	for(uint s = n_items >> 1; s > 0; s >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if(lid < s) {
			uint i = dp*(2*lid+1)-1;
			uint j = dp*(2*lid+2)-1;
			b[j] += b[i];
		}

		dp <<= 1;
	}

	if(lid == 0)
		b[n_items - 1] = 0;

	for(uint s = 1; s < n_items; s <<= 1) {
		dp >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if(lid < s) {
			uint i = dp*(2*lid+1)-1;
			uint j = dp*(2*lid+2)-1;

			float t = b[j];
			b[j] += b[i];
			b[i] = t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
#endif
	//////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////

	return;
}
	
