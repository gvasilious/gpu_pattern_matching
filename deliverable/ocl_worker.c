#include <stdio.h>
#include <stdlib.h>

#include "common.h"
#include "clutil.h"
#include "acsmx.h"
#include "databuf.h"
#include "ocl_aho_match.h"
#include "ocl_worker.h"
#include "utils.h"


/*
 * creates a new worker context
 */
struct ocl_worker_ctx *
ocl_worker_ctx_create(int dev_pos)
{
	struct ocl_worker_ctx *ocl_w_ctx;

	/* malloc the context */
	ocl_w_ctx = NULL;
	ocl_w_ctx = MALLOC(sizeof(struct ocl_worker_ctx));
	if (!ocl_w_ctx)
		return NULL;

	/* create and initialize the OpenCL context */
	clsetupf(&ocl_w_ctx->cl, "ahomatch.cl", "ahomatch", NULL, dev_pos, -1);

	return ocl_w_ctx;
}


/*
 * initializes a new worker context
 */
int
ocl_worker_ctx_init(struct ocl_worker_ctx *ocl_w_ctx, int dev_pos,
    size_t local_ws, size_t global_ws, int mapped, char *pat_path, int hex_pat, 
    int pat_size_limit, size_t max_chunk_size, int max_results, int verbose,
    int id, int thread_no, int total_files, int *fds, char **filenames)
{
	int i;
	FILE *pfp;
	char pattern[MAX_PAT_SIZE];

	/* read the pattern file and make the serialized DFA, copy to device */
	ocl_w_ctx->acsm = acsm_new();
	if ((pfp = fopen(pat_path, "r")) == NULL) {
		return -1;
	}

	i = 0;
	while (fgets(pattern, sizeof(pattern), pfp)) {
		/* strip ending newlines */
		if (pattern[strlen(pattern) - 1] == '\n')
			pattern[strlen(pattern) - 1] = '\0';

		if (hex_pat) {
			if (pat_size_limit != -1)
				pattern[pat_size_limit * 2] = '\0';
			acsm_add_pattern(ocl_w_ctx->acsm,
			    printable_hex_to_bytes((unsigned char *)pattern),
			    strlen(pattern) / 2,  0, 0, 0, 0, i);
		} else {
			if (pat_size_limit != -1)
				pattern[pat_size_limit] = '\0';
			acsm_add_pattern(ocl_w_ctx->acsm,
			    (unsigned char *)pattern,
			    strlen(pattern), 0, 0, 0, 0, i);
		}
		i++;
	}
	fclose(pfp);
	acsm_compile(ocl_w_ctx->acsm);
	acsm_gen_state_table(ocl_w_ctx->acsm, mapped, ocl_w_ctx->cl.ctx,
	    ocl_w_ctx->cl.queue);
	acsm_cleanup(ocl_w_ctx->acsm);

	/* create a new data buffer */
	ocl_w_ctx->db = databuf_new(global_ws, max_chunk_size, max_results,
	    mapped, ocl_w_ctx->cl.ctx, ocl_w_ctx->cl.queue);
	
	/* init OpenCL worker context variables */
	ocl_w_ctx->local_ws    = local_ws;
	ocl_w_ctx->global_ws   = global_ws;
	ocl_w_ctx->matches     = 0;
	ocl_w_ctx->bytes       = 0;
	ocl_w_ctx->rounds      = 0;
	ocl_w_ctx->verbose     = verbose;
	ocl_w_ctx->id          = id;
	ocl_w_ctx->thread_no   = thread_no;
	ocl_w_ctx->total_files = total_files;
	ocl_w_ctx->fds         = fds;
	ocl_w_ctx->filenames   = filenames;

	return 0;
}


/*
 * frees the worker context
 */
void
ocl_worker_ctx_free(struct ocl_worker_ctx *ctx)
{
	databuf_free(ctx->db, ctx->db->mapped, ctx->cl.queue);
	acsm_free(ctx->acsm);
	FREE(ctx);

	return;
}
