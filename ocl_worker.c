#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <ctype.h>
#include "common.h"
#include "acsmx.h"
#include "databuf.h"
#include "ocl_aho_match.h"
#include "ocl_prefix_sum.h"
#include "ocl_compact_array.h"
#include "ocl_context.h"
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
	clinitctx(&ocl_w_ctx->cl, dev_pos, -1);

	ocl_aho_match_init(&ocl_w_ctx->cl);

	ocl_prefix_sum_init(&ocl_w_ctx->cl);

	ocl_compact_array_init(&ocl_w_ctx->cl);

	return ocl_w_ctx;
}


/*
 * initializes a new worker context
 */
int
ocl_worker_ctx_init(struct ocl_worker_ctx *ocl_w_ctx, int dev_pos,
    size_t local_ws, size_t global_ws, int mapped, char *pat_path, int hex_pat, 
    int pat_size_limit, size_t max_chunk_size, int max_results, int verbose,
    int text_mode, int follow, int id, int thread_no, int total_files, int *fds,
    char **filenames)
{
	int i, j;
	long int pat_id;
	char *ptr;
	FILE *pfp;
	char line[MAX_PAT_SIZE];
	char *pattern;
	unsigned int pattern_len;
	char pid[MAX_PAT_SIZE];
	int categ = 0; /* categorical format means patterns
			  are in the form "[ID] [PATTERN]", where ID is int */

	/* read the pattern file and make the serialized DFA, copy to device */
	ocl_w_ctx->acsm = acsm_new();

	/* open file with patterns */
	if ((pfp = fopen(pat_path, "r")) == NULL) {
		return -1;
	}

	i = 0;
	while (fgets(line, sizeof(line), pfp)) {
		/* strip ending newlines */
		if (line[strlen(line) - 1] == '\n')
			line[strlen(line) - 1] = '\0';

		if (i == 0) { /* if first line, check for the format */
			j = 0;
			categ = 0;

			/* find the first space */
			for (j=0; i < strlen(line); j++) {
				if (line[j] == ' ' || line[j] == '\t') {
					j--;
					break;
				}
			}

			/* go back to the start and check if everything is a digit */
			for (j; j > 0; j--) {
				if (!isdigit(line[j]))
					break;
			}

			/* final check for the first one; it can also be a '+' or a '-' */
			if (j == 0) {
			       if (line[0] == '+' || line[0] == '-' || isdigit(line[0]))
				       categ = 1;
			}
		}

		if (categ == 1) {
			errno = 0;
			pat_id = strtol(line, &ptr, 10);
	
			if ((errno == ERANGE && (pat_id == LONG_MAX || pat_id == LONG_MIN))
		     			|| (errno != 0 && pat_id == 0)) {
				return -1;
			}
	
			/* eat white spaces */
			while (isspace(*ptr) && ptr != &line[MAX_PAT_SIZE - 1])
				ptr++;

			pattern = ptr;
			pattern_len = strlen(pattern);
		} else {
			pattern = line;
			pattern_len = strlen(pattern);
			pat_id = i;
		}

		if (pattern[0] == '"' && pattern[pattern_len - 1] == '"') {
			pattern[pattern_len - 1] = '\0';
			pattern = &pattern[1];
			pattern_len -= 2;
		}

		if (hex_pat) {
			if (pat_size_limit != -1)
				pattern[pat_size_limit * 2] = '\0';
			acsm_add_pattern(ocl_w_ctx->acsm,
			    printable_hex_to_bytes((unsigned char *)pattern),
			    strlen(pattern) / 2,  0, 0, 0, 0, pat_id);
		} else {
			if (pat_size_limit != -1)
				pattern[pat_size_limit] = '\0';
			acsm_add_pattern(ocl_w_ctx->acsm,
			    (unsigned char *)pattern,
			    strlen(pattern), 0, 0, 0, 0, pat_id);
		}
		i++;
	}
	fclose(pfp);

	/* compile added patterns to a state machine */
	acsm_compile(ocl_w_ctx->acsm);

	/* generate a serialized state machine and load it to the device */
	acsm_gen_state_table(ocl_w_ctx->acsm, mapped, ocl_w_ctx->cl.ctx,
	    ocl_w_ctx->cl.queue);

	/* get the table with all patterns and their metadata */
	ocl_w_ctx->patterns = acsm_get_patterns_table(ocl_w_ctx->acsm);

	ocl_w_ctx->patterns_size = ocl_w_ctx->acsm->num_patterns;

	/* cleanup to save some space */
	acsm_cleanup(ocl_w_ctx->acsm);

	/* create a new data buffer */
	ocl_w_ctx->db = databuf_new(global_ws, max_chunk_size, max_results,
	    mapped, &ocl_w_ctx->cl);
	
	/* init OpenCL worker context variables */
	ocl_w_ctx->local_ws         = local_ws;
	ocl_w_ctx->global_ws        = global_ws;
	ocl_w_ctx->matches_total    = 0;
	ocl_w_ctx->matches_reported = 0;
	ocl_w_ctx->bytes            = 0;
	ocl_w_ctx->lines            = 0;
	ocl_w_ctx->rounds           = 0;
	ocl_w_ctx->verbose          = verbose;
	ocl_w_ctx->text_mode        = text_mode;
	ocl_w_ctx->follow           = follow;
	ocl_w_ctx->id               = id;
	ocl_w_ctx->thread_no        = thread_no;
	ocl_w_ctx->total_files      = total_files;
	ocl_w_ctx->fds              = fds;
	ocl_w_ctx->filenames        = filenames;

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
