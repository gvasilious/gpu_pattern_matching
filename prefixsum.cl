// Copyright (c) 2014 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly
//__attribute__((reqd_work_group_size(256, 0, 0)))

__kernel void Calc_wg_offsets_naive(
                            __global const uint* gHistArray,
                            __global uint* gPrefixsumArray,
                            uint bin_size
                            )
{
    uint lid = get_local_id(0);
    uint binId = get_group_id(0);

    //calculate source/destination offset for workgroup
    uint group_offset = binId * bin_size;
    local uint maxval;

    //initialize cumulative prefix
    if( lid == 0 )  maxval = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    do
    {
        //perform a scan for every workitem
        uint prefix_sum=0;
        for(int i=0; i<lid; i++)
            prefix_sum += gHistArray[group_offset + i];
        
        //store result
        gPrefixsumArray[group_offset + lid] = prefix_sum + maxval;
        prefix_sum += gHistArray[group_offset + lid];

        //update group offset and cumulative prefix
        if( lid == get_local_size(0)-1 )  maxval += prefix_sum;
        barrier(CLK_LOCAL_MEM_FENCE);

        group_offset += get_local_size(0);
    }
    while(group_offset < (binId+1) * bin_size);
}

#define WARP_SHIFT 4
#define GRP_SHIFT 8
#define BANK_OFFSET(n)     ((n) >> WARP_SHIFT + (n) >> GRP_SHIFT)

__kernel void Calc_wg_offsets_Blelloch(__global const uint* gHistArray,
                              __global uint* gPrefixsumArray,
                              uint bin_size,
			      int max_results,
                              __local uint* temp
                            )
{
    int lid = get_local_id(0);
    uint binId = get_group_id(0);
    int n = get_local_size(0) * 2;

    uint group_offset = binId * bin_size;
    uint maxval = 0;
    do
    {
        // calculate array indices and offsets to avoid SLM bank conflicts
        int ai = lid;
        int bi = lid + (n>>1);  
        int bankOffsetA = BANK_OFFSET(ai);
        int bankOffsetB = BANK_OFFSET(bi);

        // load input into local memory
        temp[ai + bankOffsetA] = gHistArray[group_offset + ai];
        temp[bi + bankOffsetB] = gHistArray[group_offset + bi];

        // parallel prefix sum up sweep phase
        int offset = 1;
        for (int d = n>>1; d > 0; d >>= 1)
        {   
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < d)
            {  
                int ai = offset * (2*lid + 1)-1;
                int bi = offset * (2*lid + 2)-1;
                ai += BANK_OFFSET(ai);
                bi += BANK_OFFSET(bi);
                temp[bi] += temp[ai];  
            }  
            offset <<= 1; 
        }

        // clear the last element
        if (lid == 0)
        {
            temp[n - 1 + BANK_OFFSET(n - 1)] = 0;
        }

        // down sweep phase
        for (int d = 1; d < n; d <<= 1)
        {  
            offset >>= 1;  
            barrier(CLK_LOCAL_MEM_FENCE);

            if (lid < d)
            {
                int ai = offset * (2*lid + 1)-1;
                int bi = offset * (2*lid + 2)-1;
                ai += BANK_OFFSET(ai);
                bi += BANK_OFFSET(bi);
         
                uint t = temp[ai];  
                temp[ai] = temp[bi];  
                temp[bi] += t;   
            }
        }  
        barrier(CLK_LOCAL_MEM_FENCE);

        //output scan result to global memory
        gPrefixsumArray[group_offset + ai] = temp[ai + bankOffsetA] + maxval;
        gPrefixsumArray[group_offset + bi] = temp[bi + bankOffsetB] + maxval;

        //update cumulative prefix sum shift and histogram index for next iteration
        maxval += temp[n - 1 + BANK_OFFSET(n - 1)] + gHistArray[group_offset + n - 1];
        group_offset += n;
    }
    while(group_offset < (binId+1) * bin_size);
}


__kernel void scan_prefixsum(__global float *a,
			    __global float *r,
			    uint n_items,
			    int max_results,
			    __local float *b
			    )
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint dp = 1;

    b[2*lid] = a[2*gid];
    b[2*lid+1] = a[2*gid+1];

    for(uint s = n_items>>1; s > 0; s >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if(lid < s) {
            uint i = dp*(2*lid+1)-1;
            uint j = dp*(2*lid+2)-1;
            b[j] += b[i];
        }

        dp <<= 1;
    }

    if(lid == 0) b[n_items - 1] = 0;

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

    r[2*gid] = b[2*lid];
    r[2*gid+1] = b[2*lid+1];
}
