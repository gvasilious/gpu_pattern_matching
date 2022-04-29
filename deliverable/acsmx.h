#ifndef _ACSMX_H_
#define _ACSMX_H_


#include <stdio.h>
#include <stdlib.h>

#include <CL/opencl.h>


/* ASCII alphabet size */
#define ALPHABET_SIZE	256

/* default Fail State */
#define ACSM_FAIL_STATE	-1


/* Aho-Corasick state machine pattern */
struct _acsm_pattern {
	struct  _acsm_pattern	*next;
	unsigned char		*pattern;
	unsigned char		*casepattern;
	int			n;
	int			nocase;
	int			offset;
	int			depth;
	void			*id;
	int			iid;
};
typedef struct _acsm_pattern acsm_pattern_t;


/* Aho-Corasick state machine state table */
struct _acsm_state_table {
	int		next_state[ALPHABET_SIZE];
	int		fail_state;
	int		num_finals;
	acsm_pattern_t	*match_list;
};
typedef struct _acsm_state_table acsm_state_table_t;


/* Aho-Corasick state machine */
struct _acsm {
	int			max_states;
	int			num_states;
	int			max_pattern_len;
	size_t			size;
	acsm_pattern_t		*patterns;
	acsm_state_table_t	*state_table;
	int			*h_trans;
	cl_mem			d_trans;
};
typedef struct _acsm acsm_t;


/*
 * creates a new Aho-Corasick state machine
 *
 * ret: a new Aho-Corasick state machine
 */
acsm_t *
acsm_new(void);


/*
 * adds a pattern to the list of patterns for this state machine
 *
 * arg0: Aho-Corasick state machine
 * arg1: a string containing the pattern
 * arg2: pattern size in bytes
 * arg3: a flag indicating case sensitivity
 * arg4: pattern offset (depricated)
 * arg4: pattern depth (depricated)
 * arg5: callback handler (depricated)
 * arg6: pattern ID
 */
void
acsm_add_pattern(acsm_t *, unsigned char *, int, int, int, int, void *, int);


/*
 * compiles the state machine
 *
 * arg0: Aho-Corasick state machine
 */
void
acsm_compile(acsm_t *);


/*
 * creates the serialized DFA state table and transfers it to the device
 *
 * arg0: Aho-Corasick state machine
 * arg1: memory mapped buffers flag
 * arg2: OpenCL context
 * arg3: OpenCL command queue
 */
void
acsm_gen_state_table(acsm_t *, int, cl_context, cl_command_queue);


/*
 * returns the size of the largest pattern
 *
 * arg0: Aho-Corasick state machine
 *
 * ret:  size of the largest pattern in bytes
 */
int
acsm_get_max_pattern_size(acsm_t *);


/*
 * returns the number of states in the automaton
 *
 * arg0: Aho-Corasick state machine
 *
 * ret:  number of states
 */
int
acsm_get_states(acsm_t *);


/*
 * returns the automaton size in bytes
 *
 * arg0: Aho-Corasick state machine
 *
 * ret:  automaton size in bytes
 */
size_t
acsm_get_size(acsm_t *);


/*
 * cleans the memory and keeps the serialized DFA state table
 *
 * arg0: Aho-Corasick state machine
 */
void
acsm_cleanup(acsm_t *);


/*
 * frees all acsm memory
 *
 * arg0: Aho-Corasick state machine
 */
void
acsm_free(acsm_t *);


#endif /* _ACSMX_H_ */
