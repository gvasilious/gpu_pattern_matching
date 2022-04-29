// Performs stream compaction of arr_src using the provided prefix exclusive sums.
// The format of the arr_src is the following:
//
//                       -------------------------------------
// #elems per column ->  | 4| 0| 5| 2| 3| 0| 0| 1| 2| 4| 2| 0|
//                       -------------------------------------
//                       | 3| 0| 4| 3| 4| 0| 0| 6| 7| 5| 4| 0|
//                       -------------------------------------
//                       | 8| 0| 9| 6| 3| 0| 0| 0| 2| 2| 4| 0|
//                       -------------------------------------
//                       | 9| 0| 8| 0| 9| 0| 0| 0| 0| 7| 0| 0|
//                       -------------------------------------
//                       | 5| 0| 2| 0| 0| 0| 0| 0| 0| 1| 0| 0|
//                       -------------------------------------
//                       | 0| 0| 7| 0| 0| 0| 0| 0| 0| 0| 0| 0|
//                       -------------------------------------
//                       | 0| 0| 0| 0| 0| 0| 0| 0| 0| 0| 0| 0|
//                       -------------------------------------
//                       | 0| 0| 0| 0| 0| 0| 0| 0| 0| 0| 0| 0|
//                       -------------------------------------
//
// The provided prefix sums is the following:
//
//                       -------------------------------------
//                       | 0| 0| 4| 4|11|14|14|14|15|17|21|23|
//                       -------------------------------------
//
// The resulting arr_dst is the following:
//
//     index:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
//            -------------------------------------------------------------------------
//     value: | 3| 8| 9| 5| 4| 9| 8| 2| 7| 3| 6| 4| 3| 9| 6| 7| 2| 5| 2| 7| 1| 4| 4| -|
//            -------------------------------------------------------------------------
//
//
// takes an array of a specific format and returns an array in which the data
// are compacted sequentially.


__kernel void
compactarray(__global int *arr_dst, __global int *arr_src, __global int *prefixsum, const int len, const int max_results) {
	int i, offset;
	int gid, gsz;
	int matches;

	gid = get_global_id(0);
	gsz = get_global_size(0);

	if (gid == 0) {
		/* last element of prefixsum contains the sum
		exlcuding the last one; hence we add it from
		arr_src */
		arr_dst[0] = prefixsum[len-1] + arr_src[len-1];
		arr_dst[ arr_dst[0] + 1 ] = arr_src[max_results * len];
	}

	if (gid >= len) {
		return;
	}

	/* get the offset to write the matches for this thread */
	offset = prefixsum[gid];

	matches = arr_src[gid];
	for (i=0; i < matches && i < max_results - 1; ++i) {
		arr_dst[offset + 1 + i] = arr_src[len*(i+1) + gid];
	}
}
