#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/param.h>
#include <CL/opencl.h>

#include "acsmx.h"
#include "common.h"
#include "databuf.h"
#include "file_traverse.h"
#include "ocl_aho_match.h"
#include "ocl_context.h"
#include "ocl_worker.h"
#include "utils.h"


int
callback_match(int, int, int, int, void*);

int terminate = 0;

void
signal_handler(int signum)
{
	terminate = 1;
}

/*
 * CPU worker thread
 */
void *
cpu_worker(void *worker_ctx)
{
	int e = 0, done = 0, cur_file = 0;
	size_t rd_bytes = 0, rd_lines = 0;
	struct ocl_worker_ctx *ctx = 0x0;
	FILE *fp = 0x0;

	ctx = (struct ocl_worker_ctx *)worker_ctx;
	if (!ctx)
		ERRX(1, "ERROR: thread has NULL context!\n");

	cur_file = ctx->id;

	/* more OpenCL worker threads than files */
	if (cur_file >= ctx->total_files)
		return 0;

	if (ctx->text_mode)
		fp = fdopen(ctx->fds[cur_file], "r");

	done     = 0;
	rd_bytes = 0;
	rd_lines = 0;
	while (!done) {
		/* check if program needs to terminate */
		if (terminate) {
			done = 1;
			goto process;
		}

		/* read current file */
		if (ctx->text_mode)
			e = databuf_add_fp(ctx->db, fp, cur_file,
					1 /* aligned */, &rd_bytes, &rd_lines);
		else
			e = databuf_add_fd(ctx->db, ctx->fds[cur_file],
		    			cur_file, &rd_bytes);

		ctx->lines += rd_lines;
		ctx->bytes += rd_bytes;

		/* current file has been read */
		if (rd_bytes == 0) {
			if (!ctx->follow && ctx->text_mode)
				fclose(fp);

			if (!ctx->follow)
				close(ctx->fds[cur_file]);

			/* proceed with the next file */
			cur_file += ctx->thread_no;

			if (ctx->text_mode)
				fdopen(ctx->fds[cur_file], "r");

			/*
			 * no more files
			 * the buffer may have data so force a last kernel
			 */
			if (cur_file >= ctx->total_files) {
				if (ctx->follow) {
					cur_file = ctx->id;
				} else {
					done = 1;
					goto process;
				}
			}

			continue;
		}

		/* the buffer can hold more data */
		if ((e != -1) && (e != -2))
		continue;

process:

		if (ctx->db->chunks > 0) {
			/* copy the data to the device */
			databuf_copy_host_to_device(ctx->db, ctx->cl.queue);

			/* scan data */
			ocl_aho_match(&(ctx->cl), ctx->db, ctx->acsm, ctx->local_ws, 1 /* stream */);

#ifdef COMPACT_RESULTS
			/* compute prefix sums; will be used to do array compaction */
			//ocl_prefix_sum(&(ctx->cl), ctx->db, ctx->db->chunks);

			/* do the array compaction */
			//ocl_compact_array(&(ctx->cl), ctx->db, /* 1024 */ ctx->local_ws); //TODO: XXX find a way to pass local_ws as a parameter
#endif
			/* get the results */
			databuf_copy_device_to_host(ctx->db, ctx->cl.queue);

			/* get the total matches */
			int callback_match(int f_id, int p_idx, int c_id, int off, void *uarg);

			ctx->matches_total += databuf_process_results(ctx->db, callback_match, ctx);

			/* reset the buffer for the next batch */
			databuf_reset(ctx->db);

			ctx->rounds++;
		}
	}

	return 0;
}


/*
 * prints the usage message
 */
void
usage(void)
{
	printf(
	    "\n"
	    "Usage:\n"
	    "    ocl_aho_grep -f file -p file -B chunk_size -D devpos\n"
	    "                 -G global_ws -L local_ws [-m max]\n"
	    "                 [-w cpu_threads] [-R max] [-tvxM]\n"
	    "    ocl_aho_grep -h\n"
	);
	printf(
	    "\n"
	    "Options:\n"
	    "  -f    file         Path to the input file.\n"
	    "                     ! The path can be a single directory, a\n"
	    "                     single file or many comma-separated files.\n"
	    "  -p    file         Path to the file containing the patterns, one\n"
	    "                     pattern per line.\n"
	    "  -F                 Process appended data as files grow. It can\n"
	    "                     be practical when needed to process data\n"
	    "                     continuously, e.g., from a FIFO.\n"
	    "  -B    chunk_size   Maximum data chunk size (in bytes), that each\n"
	    "                     OpenCL kernel thread will process.\n"
	    "  -D    devpos       A number indicating which OpenCL device will.\n"
	    "                     be used.\n"
	    "                     ! Device positions can be found with clinfo.\n"
	    "  -G    global_ws    Global Work Size for the OpenCL kernel.\n"
	    "                     ! buffer_size = global_ws * chunk_size.\n"
	    "  -L    local_ws     Local Work Size for the OpenCL kernel.\n"
	    "  -m    max          Limit the pattern size to max bytes.\n"
	    "                     ! If a pattern is smaller than max, the\n"
	    "                     entire pattern will be used.\n"
	    "  -w    cpu_threads  Number of CPU threads that will be used for\n"
	    "                     feeding OpenCL kernel with data.\n"
	    "                     ! Default: 2.\n"
	    "  -R    max          Maximum number of result slots per chunk.\n"
	    "                     ! The first is always reserved in order to\n"
	    "                     store the number of matches found per chunk.\n"
	    "                     The rest are used to store the offsets where\n"
	    "                     the patterns have been found. Default: 16.\n"
	    "  -v                 Prints the file name and the patterns found.\n"
	    "                     ! The number of pattern IDs reported is\n"
	    "                     affected by [-R max].\n"
	    "  -t                 Treats input files as text files; tries to\n"
	    "                     ! read line-wise whenever possible.\n"
	    "  -x                 Handles the patterns as printable hex.\n"
	    "                     ! The patterns should not contain the '0x'\n"
	    "                     notation.\n"
            "  -M                 Set mapped buffers (CPU or integrated GPU).\n"
	    "                     ! Default: 0.\n"
	    "  -h                 This help message.\n"
	);
	exit(EXIT_FAILURE);
}


/*
 * checks the user's arguments
 */
void
check_args(char *pat_path, char *file_path, int dev_pos, size_t global_ws,
    size_t local_ws, size_t max_chunk_size, int thread_no, int pat_size_limit,
    int max_results)
{
	int err;

	err = 0;

	if (!pat_path) {
		printf("ERROR: No pattern file\n");
		err++;
	} else if (!file_exists(pat_path)) {

		printf("ERROR: File '%s' does not exist\n", pat_path);
		err++;
	}
	if (!file_path) {
		printf("ERROR: No data file\n");
		err++;
	}
	if (dev_pos == -1) {
		printf("ERROR: No device position\n");
		err++;
	}
	if (global_ws == -1) {
		printf("ERROR: No global work size\n");
		err++;
	}
	if (local_ws == -1) {
		printf("ERROR: No local work size\n");
		err++;
	}
	if (max_chunk_size == -1) {
		printf("ERROR: No maximum chunk size\n");
		err++;
	}
	if (thread_no <= 0) {
		printf("ERROR: The thread number must be greater than 0\n");
		err++;
	}
	if ((pat_size_limit != -1) && (pat_size_limit <= 0)) {
		printf("ERROR: The pattern size limit should be >= 1\n");
		err++;
	}
	if (pat_size_limit >= MAX_PAT_SIZE) {
		printf("ERROR: The pattern size limit should be <= %d\n",
		    MAX_PAT_SIZE - 1);
		err++;
	}
	if (max_results <= 0) {
		printf("ERROR: The maximum result cells should be >= 1\n");
		err++;
	}

	if (err)
		usage();
}

/*
 * Print details for each match found
 */
int callback_match(int f_id, int p_idx, int c_id, int off, void *uarg) {
	int i;
	struct ocl_worker_ctx *ctx = (struct ocl_worker_ctx*)uarg;

	int pat_id    = ctx->patterns[p_idx].iid;
	unsigned char *pat_name  = ctx->patterns[p_idx].pattern;
	int pat_len   = ctx->patterns[p_idx].n;
	char *fname   = ctx->filenames[ctx->db->file_ids[c_id]];
	int off_rel   = 0;

	ctx->matches_reported += 1;

	if (ctx->verbose) {
		off_rel = off - ctx->db->h_indices[c_id];
		printf("Pattern %d ('%s') found in file '%s' at offset %d [relative: %d]\n",
				pat_id, pat_name, fname, off, off_rel);

		if (ctx->text_mode) {
			for (i = 0; i < ctx->db->h_sizes[c_id]; i++) {
				printf("%c", ctx->db->h_data[/*ctx->db->h_indices[c_id] + */i]);
			}
		} else {
			printf(" ... ");
			/* XXX off points to the end of pattern, not the start */
			for (i = MAX(0, off - 10) ; i < off + pat_len + 10; i++) {
				if (i >= ctx->db->size ||
						ctx->db->h_data[/*ctx->db->h_indices[c_id] + */i] == '\n') {
					break;
				}
				printf("%c", ctx->db->h_data[/*ctx->db->h_indices[c_id] + */i]);
			}
			printf(" ... \n");
		}
	}

	return 0;
}

/*
 * checks if the input parameters are aligned
 * changes those who are not
 * returns how mane got changed
 */
int
align_parameters(size_t *local_ws, size_t *global_ws, size_t *max_chunk_size)
{
	int e;

	e = 0;
	if (*local_ws % 16) {
		printf("WARNING: local work size '%lu' is not 16B aligned. ",
		    *local_ws);
		*local_ws = ROUNDUP(*local_ws, 16);
		printf("Will use '%lu' instead\n", *local_ws);
		e++;
	}

	if (*global_ws % 16) {
		printf("WARNING: global work size %lu is not 16B aligned. ",
		    *global_ws);
		*global_ws = ROUNDUP(*global_ws, 16);
		printf("Will use '%lu' instead.\n", *global_ws);
		e++;
	}

	if (*max_chunk_size % 16) {
		printf("WARNING: max chunk size '%lu' is not 16B aligned. ",
		    *max_chunk_size);
		*max_chunk_size = ROUNDUP(*max_chunk_size, 16);
		printf("Will use '%lu' instead.\n", *max_chunk_size);
		e++;
	}

	return e;
}



/*
 * a multi-threaded Aho-Corasick matcher with OpenCL support
 */
int
main(int argc, char *argv[])
{
	int i;
	int e;				/* error code handle                  */
	int opt;			/* argument parsing option            */
	int dev_pos;			/* device position (clinfo)           */
	int mapped;			/* memory mapped buffers flag         */
	int hex_pat;			/* printable hex patterns flag        */
	int verbose;			/* verbosity flag                     */
	int text_mode;			/* try to read input files line-wise  */
	int follow;			/* process appended data as files grow*/
	int total_rounds;		/* processing rounds                  */
	int thread_no;			/* number of POSIX threads            */
	int total_files;		/* number of files processed          */
	int o_files;			/* number of files opened             */
	int *fds;			/* file descriptor array              */
	int max_results;		/* maxm number of results per chunk   */
	int pat_size_limit;		/* maximum pattern size limit         */
	size_t total_matches;		/* total matches found                */
	size_t reported_matches;	/* number of matches reported         */
	size_t total_bytes;		/* total bytes processed              */
	size_t total_lines;		/* total lines of text processed      */
	size_t global_ws;		/* global work size                   */
	size_t local_ws;		/* local work size                    */
	size_t max_chunk_size;		/* maximum data per thread (bytes     */
	char *file;			/* a dummy for strtok                 */
	char *pat_path;			/* path to pattern file               */
	char *data_path;		/* path to input file(s)              */
	char *reg_files;		/* path to input every file in a dir  */
	char **filenames;		/* string array with the filenames    */
	size_t start_time;		/* starting time end-to-end           */
	size_t end_time;		/* ending time end-to-end             */
	size_t e2e_time;		/* end-to-end time in usecs           */
	struct ocl_worker_ctx **w_ctx;	/* OpenCL worker contexts array       */
	pthread_t *threads;		/* thread handles                     */
	struct rlimit rlim;		/* resource limits                    */


	/* initialize */
	dev_pos        = -1;
	mapped         = 0;
	global_ws      = -1;
	local_ws       = -1;
	max_chunk_size = -1;
	pat_size_limit = -1;
	pat_path       = NULL;
	data_path      = NULL;
	verbose        = 0;
	text_mode      = 0;
	follow         = 0;
	hex_pat        = 0;
	thread_no      = 2;
	threads        = NULL;
	max_results    = MAX_RESULTS;


	/* get options */
	while ((opt = getopt(argc, argv, "f:m:p:tw:vxB:D:FG:L:R:Mh")) != -1) {
		switch (opt) {
		case 'f':
			data_path = strdup(optarg);
			break;
		case 'm':
			pat_size_limit = atoi(optarg);
			break;
		case 'p':
			pat_path = strdup(optarg);
			break;
		case 't':
			text_mode = 1;
			break;
		case 'w':
			thread_no = atoi(optarg);
			break;
		case 'v':
			verbose = 1;
			break;
		case 'x':
			hex_pat = 1;
			break;
		case 'B':
			max_chunk_size = atol(optarg);
			break;
		case 'D':
			dev_pos = atoi(optarg);
			break;
		case 'F':
			follow = 1;
			break;
		case 'G':
			global_ws = atol(optarg);
			break;
		case 'L':
			local_ws = atol(optarg);
			break;
		case 'R':
			max_results = atoi(optarg);
			break;
		case 'M':
			mapped = 1;
			break;
		case 'h':
		default:
			usage();
		}
	}


	/* Get the number of maximum open file descriptors */
	if (getrlimit(RLIMIT_NOFILE, &rlim) == -1) {
		ERRX(1, "ERROR: getrlimit RLIMIT_NOFILE\n");
	}

	/* Expand soft limit to the maximum */
	if (rlim.rlim_cur < rlim.rlim_max) {
		rlim.rlim_cur = rlim.rlim_max;
		if (setrlimit(RLIMIT_NOFILE, &rlim) == -1)
			ERRX(1, "ERROR: setrlimit RLIMIT_NOFILE\n");
	}


	/* check arguments */
	check_args(pat_path, data_path, dev_pos, global_ws, local_ws,
	    max_chunk_size, thread_no, pat_size_limit, max_results);


	/*
	 * check the local work size, global work size and maximum bytes per 
	 * thread for missalignments
	 * fix if any
	 */
	e = 0;
	e = align_parameters(&local_ws, &global_ws, &max_chunk_size);

	printf("Local Work Size:  %lu\n", local_ws);
	printf("Global Work Size: %lu\n", global_ws);
	printf("Max Chunk Size:   %lu\n", max_chunk_size);
	printf("\n");


	/* create the OpenCL worker contexts */
	w_ctx = MALLOC(thread_no * sizeof(struct ocl_worker_ctx *));
	if (!w_ctx)
		ERRX(1, "ERROR: malloc ocl_worker_ctx\n");
	for (i = 0; i < thread_no; i++) {
		w_ctx[i] = ocl_worker_ctx_create(dev_pos);
		if (!w_ctx[i])
			ERRX(1, "ERROR: create_ocl_worker\n");
	}


	/*
	 * the path may point to a directory, a single file,
	 * or multiple comma-seperated files
	 */
	if (is_directory(data_path)) {
		reg_files = get_all_regular_files(data_path);
		FREE(data_path);
		data_path = reg_files;
	}

	/* get total files in path */
	total_files = 1;
	if (data_path[strlen(data_path) -1 ] == ',')
		data_path[strlen(data_path) - 1] = '\0';

	for (i = 0; i < strlen(data_path); ++i)
		if (data_path[i] == ',')
			total_files++;

	if (rlim.rlim_cur != RLIM_INFINITY
	    && (total_files > rlim.rlim_cur))
		ERRXV(1, "ERROR: The number of input files is larger than the\n"
		    "number of maximum open files\n"
		    "(Requested to open %u files, while the max number "
		    "is %lu).\n", total_files, rlim.rlim_cur);

	/* allocate an array for all file descriptors */
	fds = MALLOC(total_files * sizeof(int));
	if (!fds)
		ERRX(1, "ERROR: malloc ioh->fds\n");

	/* allocate array for all file names */
	filenames = MALLOC(total_files * sizeof(char *));
	if (!fds)
		ERRX(1, "ERROR: malloc ioh->filenames\n");

	/* open all files in advance */
	file = strtok(data_path, ",");
	o_files = 0;
	for (i = 0; (file != NULL) && (i < total_files); ++i) {
		if (is_regular_file(file) || is_fifo(file)) {
			if ((fds[o_files] = open(file, O_RDONLY)) == -1) {
				ERRXV(1, "ERROR: could not open '%s'\n", file);
			} else {
				filenames[o_files] = strdup(file);
				o_files++;
			}
		}
		file = strtok(NULL, ",");
	}
	total_files = o_files;

	if (!total_files)
		ERRX(1, "ERROR: Could not open input file(s) for reading.\n");

	/* initialize the OpenCL worker contexts */
	for (i = 0; i < thread_no; i++) {
		if (ocl_worker_ctx_init(w_ctx[i], dev_pos, local_ws, global_ws,
		    mapped, pat_path, hex_pat, pat_size_limit, max_chunk_size,
		    max_results, verbose, text_mode, follow, i, thread_no,
		    total_files, fds, filenames) != 0) {
			ERRX(1, "ERROR: init_ocl_worker_ctx\n");
		}
	}


	/* allocate thread handles */
	threads = calloc(thread_no, sizeof(pthread_t));
	if (!threads)
		ERRX(1, "ERROR: calloc threads\n");


	signal(SIGINT, signal_handler);

	/* spawn the OpenCL worker threads */
	start_time = gettime();
	for (i = 0; i < thread_no; ++i) {
		if (pthread_create(&threads[i], NULL, cpu_worker,
		    (void *)w_ctx[i]) != 0) 
			ERRXV(1, "ERROR: creating thread: %d\n", i);
	}


	/* join the OpenCL worker threads */
	for (i = 0; i < thread_no; ++i) {
		e = pthread_join(threads[i], NULL);
		/*
		 * some threads may have finished
		 * before the call of pthread_join
		 */
		if ((e != 0) && (e != ESRCH))
			ERRXV(1, "ERROR: pthread_join: %d thread: %d\n", e, i);
	}
	end_time = gettime();


	/* stats */
	total_matches     = 0;
	reported_matches  = 0;
	total_bytes       = 0;
	total_lines       = 0;
	total_rounds      = 0;
	for (i = 0; i < thread_no; ++i) {
		total_matches     += w_ctx[i]->matches_total;
		reported_matches  += w_ctx[i]->matches_reported;
		total_bytes       += w_ctx[i]->bytes;
		total_lines       += w_ctx[i]->lines;
		total_rounds      += w_ctx[i]->rounds;
	}
	e2e_time = end_time - start_time;
	printf("-------------- STATS --------------\n");
	printf("Matches:             %lu\n",  total_matches);
	printf("Matches reported:    %lu\n",  reported_matches);
	printf("Time (secs):         %.5f\n", (double)e2e_time / 1000000);
	printf("Automaton states:    %d\n",
	    acsm_get_states(w_ctx[0]->acsm));
	printf("Automaton size (MB): %.3f\n",
	    (double)acsm_get_size(w_ctx[0]->acsm) / 1048576);
	printf("Processed bytes:     %lu\n",  total_bytes);
	if (total_lines)
		printf("Processed lines:     %lu\n",  total_lines);
	printf("Processed files:     %d\n",   total_files);
	printf("Kernel launches:     %d\n",   total_rounds);
	printf("Throughput (Mbps):   %.3f\n",
	    (double)((double)(total_bytes * 8) / 1048576) /
	    ((double)e2e_time / 1000000));
	printf("-----------------------------------\n\n");


	/* clean up */
	FREE(data_path);
	FREE(pat_path);
	for (i = 0; i < thread_no; ++i)
		ocl_worker_ctx_free(w_ctx[i]);
	for (i = 0; i < total_files; ++i)
		FREE(filenames[i]);
	FREE(filenames);
	FREE(fds);


	/* END */
	return 0;
}
