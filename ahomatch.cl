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
	int id, lid;
	int index;
	int size;
	int matches = 0; // count the matches per thread
	long state, state_prev;
	unsigned char c;
	unsigned char *p_c16;
	uint4 c16;

	id  = get_global_id(0);
	lid = get_local_id(0);

	matches = 0;
	// XXX no need to reset the whole array;
	//for (i = 0; i < max_results; i++) {
	//	results[chunks * i + id] = -1;
	//	//results2[max_results * id + i] = -1;
	//}

	if (id >= chunks)
		goto end;

	index = indices[id];
	size = sizes[id];

	/*
	 * The first thread is assigned with the state of the last tread of 
	 * the previous kernel call so we can grep the matches splitted over 
	 * two data buffers
	 */
	if (id == 0) //XXX check also if stream mode is on
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
					results[matches * chunks + id] =
					    *(trans +
					    (unsigned long)(ALPHABET_SIZE * 2) *
					    (unsigned long)(state_prev) +
					    (unsigned long)c +
					    (unsigned long)ALPHABET_SIZE);
					results2[matches * chunks + id] = index + i * sizeof(uint4) + j; // add index for absolute offset
				}
			}
		}
	}


	/* stream mode
	 * if stream mode is off goto end;
	 * otherwise continue
	 */

	/*
	 * The last thread saves its state for the first thread of the next
	 * bufferand returns. This will grep the matches splitted over two data
	 * buffers. This is for stream mode.
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
	 * different threads. This is for stream mode.
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
					results[matches * chunks + id] =
					    *(trans +
					    (unsigned long)(ALPHABET_SIZE * 2) *
					    (unsigned long)(state_prev) +
					    (unsigned long)c +
					    (unsigned long)ALPHABET_SIZE);
					results2[matches * chunks + id] = index + i * sizeof(uint4) + j; // add index for absolute offset
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
	results[id] = matches;
	results2[id] = matches;

	return;
}
	
