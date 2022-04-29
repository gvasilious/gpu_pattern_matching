__kernel void
find_pos(__global int*arr, const unsigned int arr_len, __global int*pos)
{
	int id = get_global_id(0);

	if (id < arr_len - 1) {
		if (arr[id] != -1 && arr[id+1] == -1) {
			*pos = id;
		}
	}

	return;
}
	
