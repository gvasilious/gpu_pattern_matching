__kernel void
ahomatch(__global int *trans, __global uint4 *data, __global int *indices, 
    __global int *sizes, __global int*results, const unsigned int chunks,
    const unsigned long data_size, const long last_state,
    const int max_pat_size, const int max_results)
{
#define ALPHABET_SIZE	256
#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

	int i;
	int j;
	int id;
	int index;
	int size;
	long state;
	unsigned char c;
	unsigned char *p_c16;
	uint4 c16;

	id = get_global_id(0);
	results[max_results * id] = 0;
	for (i = 1; i < max_results; i++)
		results[max_results * id + i] = -1;
	if (id >= chunks)
		return;

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

			state = *(trans + (unsigned long)(ALPHABET_SIZE * 2) *
			    (unsigned long)state + (unsigned long)c);

			/* match */
			if (state < 0) {
				results[max_results * id]++;
				state = -state;
				if (results[max_results * id] < max_results) {
					results[max_results * id +
					    results[max_results * id]] = 
					    *(trans +
					    (unsigned long)(ALPHABET_SIZE * 2) *
					    (unsigned long)(state - 1) +
					    (unsigned long)c +
					    (unsigned long)ALPHABET_SIZE); 
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
		return;
	}


	/*
	 * All threads except for the last one reach this point. If their last 
	 * state is zero, they do not have a match in progress so they return
	 */
	if (state == 0)
		return;


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
			return;

		c16 = *(data + index / sizeof(uint4) + i);
		p_c16 = (unsigned char *)&c16;

		/* loop on the fetched data */
		for (j = 0; j < sizeof(uint4); j++) {
			c = p_c16[j];

			state = *(trans + (unsigned long)(ALPHABET_SIZE * 2) *
			    (unsigned long)state + (unsigned long)c);

			/*
			 * If the continued match fails, return here so 
			 * this thread does not grep duplicate matches
			 */
			if (state == 0)
				return;

			/* match */
			if (state < 0) {
				results[max_results * id]++;
				state = -state;
				if (results[max_results * id] < max_results) {
					results[max_results * id +
					    results[max_results * id]] = 
					    *(trans +
					    (unsigned long)(ALPHABET_SIZE * 2) *
					    (unsigned long)(state - 1) +
					    (unsigned long)c +
					    (unsigned long)ALPHABET_SIZE); 
				}
				return;
			}
		}
	}

	return;
}
	
