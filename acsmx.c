/*
 * Multi-Pattern Search Engine
 *
 * Aho-Corasick State Machine -  uses a Deterministic Finite Automata - DFA
 *
 * Copyright (C) 2002 Sourcefire,Inc.
 * Marc Norton
 *
 *  
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 2 as
 * published by the Free Software Foundation. You may not use, modify or
 * distribute this program under any other version of the GNU General
 * Public License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 *   Reference - Efficient String matching: An Aid to Bibliographic Search
 *               Alfred V Aho and Margaret J Corasick
 *               Bell Labratories 
 *               Copyright(C) 1975 Association for Computing Machinery,Inc
 *
 *   Implemented from the 4 algorithms in the paper by Aho & Corasick
 *   and some implementation ideas from 'Practical Algorithms in C'
 *
 *   Notes:
 *     1) This version uses about 1024 bytes per pattern character - heavy on
 *        the memory. 
 *     2) This algorithm finds all occurrences of all patterns within a
 *        body of text.
 *     3) Support is included to handle upper and lower case matching.
 *     4) Some comopilers optimize the search routine well, others don't, this
 *        makes all the difference.
 *     5) Aho inspects all bytes of the search text, but only once so it's very
 *        efficient, if the patterns are all large than the Modified Wu-Manbar
 *        method is often faster.
 *     6) I don't subscribe to any one method is best for all searching needs,
 *        the data decides which method is best, and we don't know until after
 *        the search method has been tested on the specific data sets.
 *        
 *
 *  May 2002: Marc Norton 1st Version  
 *  Jun 2002: Modified interface for SNORT, added case support
 *  Aug 2002: Cleaned up comments, and removed dead code.
 *  Nov 2002: Fixed queue_init(), added count = 0
 *
 *  author:	Dimitris Deyannis
 *  contact:	deyannis@ics.forth.gr
 *  changes by author:
 *  	Sep 2016: changed the (horrible whatever it was) style to OpenBSD
 *	          style(9)
 *  	Oct 2016: major refactoring, added dump to .bin, added iid table dump,
 *	          removed dead code
 *	Nov 2016: changed functions returning just zero to void, changed
 *	          declaring and initializing variables at the same time,
 *	          partially documented the code with comments for the newcomers
 *  	Mar 2017: added transition table to OpenCL buffers, removed dumping to 
 *	          file for current version, removed iid table dump
 *
 *  author:	Giorgos Vasiliadis
 *  contact:	gvasil@ics.forth.gr
 *  changes by author:
 *      Dec 2019: added function to create a stand-alone table for all patterns
 *                that have been added to a state machine. The table also
 *                links any patterns that have a common prefix.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <CL/opencl.h>

#include "acsmx.h"
#include "common.h"
#include "utils.h"


#define MEMASSERT(p, s) do {						\
	if (!p) {							\
		fprintf(stderr, "ERROR! out off memory: %s!\n", s);	\
		exit(0);						\
	}								\
} while (0)


/* ================================= queue ================================== */


/*
 * simple queue_t node
 */ 
struct _qnode {
	int		state;
	struct _qnode	*next;
};
typedef struct _qnode qnode_t;


/*
 * simple queue
 */ 
struct _queue {
	qnode_t	*head;
	qnode_t	*tail;
	int	count;
};
typedef struct _queue queue_t;


/* ======================== memory handling wrappers ======================== */


/*
 * calloc wrapper
 */ 
static void *
ac_malloc(int n) {
	void *p;

	p = calloc(1, n);

	return p;
}


/*
 * free wrapper
 */ 
static void
ac_free(void *p) {
	if (p)
		free (p);

	return;
}


/* ================================= queue ================================== */


/*
 * initializes a new queue_t
 */ 
static void
queue_init(queue_t *s) 
{
	s->head = s->tail = 0;
	s->count = 0;

	return;
}


/*
 * adds a tail item to queue
 */
static void
queue_add(queue_t *s, int state) 
{
	qnode_t * q;
	if (!s->head) {
		q = s->tail = s->head = (qnode_t *)ac_malloc(sizeof(qnode_t));
		MEMASSERT(q, "queue_add");
		q->state = state;
		q->next = 0;
	} else 	{
		q = (qnode_t *)ac_malloc(sizeof(qnode_t));
		MEMASSERT(q, "queue_add");
		q->state = state;
		q->next = 0;
		s->tail->next = q;
		s->tail = q;
	}
	s->count++;

	return;
}


/*
 * removes a head item from queue
 */ 
static int
queue_remove(queue_t *s) 
{
	int state = 0;
	qnode_t *q;

	if (s->head) {
		q = s->head;
		state = q->state;
		s->head = s->head->next;
		s->count--;
		if (!s->head) {
			s->tail = 0;
			s->count = 0;
		}
		ac_free(q);
	}

	return state;
}


/*
 * returns the queue count
 */ 
static int
queue_count(queue_t *s) 
{
	return s->count;
}


/*
 * frees the queue
 */ 
static void
queue_free(queue_t *s) 
{
	while (queue_count(s))
		queue_remove(s);

	return;
}


/*
 * case Translation Table 
 */ 
static unsigned char xlatcase[256];


/* ========================== acsm build internals ========================== */


/*
 * converts the case to upper 
 */ 
static void
init_xlatcase(void) 
{
	int i;

	for (i = 0; i < ALPHABET_SIZE; i++)
		xlatcase[i] = (unsigned char)toupper(i);

	return;
}


/*
 * makes a copy with converted case
 * plz don't use this, i killed it
 */ 
static inline void
convert_case_ex(unsigned char *d, unsigned char *s, int m) 
{
	int i;

	for (i = 0; i < m; i++)
		d[i] = s[i];
		/* d[i] = xlatcase[s[i]]; */

	return;
}


/*
 * copies a match list entry
 */ 
static acsm_pattern_t *
copy_match_list_entry(acsm_pattern_t *px) 
{
	acsm_pattern_t *p;

	p = (acsm_pattern_t *)ac_malloc(sizeof(acsm_pattern_t));
	MEMASSERT(p, "copy_match_list_entry");
	memcpy(p, px, sizeof(acsm_pattern_t));
	p->next = 0;

	return p;
}


/*
 * adds a pattern to the list of patterns terminated at this state
 * insert at front of list
 */ 
static void
add_match_list_entry(acsm_t *acsm, int state, acsm_pattern_t *px) 
{
	acsm_pattern_t *p;

	p = (acsm_pattern_t *)ac_malloc(sizeof(acsm_pattern_t));
	MEMASSERT(p, "add_match_list_entry");
	memcpy(p, px, sizeof(acsm_pattern_t));
	p->next = acsm->state_table[state].match_list;
	acsm->state_table[state].match_list = p;
	acsm->state_table[state].num_finals++;

	return;
}


/* 
 * adds pattern states
 */ 
static void
add_pattern_states(acsm_t *acsm, acsm_pattern_t *p) 
{
	unsigned char *pattern;
	int state;
	int next;
	int n;

	n = p->n;
	pattern = p->pattern;
	state = 0;

	/* match up pattern with existing states */ 
	for (; n > 0; pattern++, n--) {
		next = acsm->state_table[state].next_state[*pattern];
		if (next == ACSM_FAIL_STATE)
			break;
		state = next;
	}

	/* add new states for the rest of the pattern bytes, 1 state per byte */
	for (; n > 0; pattern++, n--) {
		acsm->num_states++;
		acsm->state_table[state].next_state[*pattern] =
		    acsm->num_states;
		state = acsm->num_states;
	}

	add_match_list_entry(acsm, state, p);

	return;
}


/*
 * builds Non-Deterministic Finite Automata
 */ 
static void
build_NFA(acsm_t *acsm) 
{
	int i;
	int r;
	int s;
	int fs;
	int next;
	queue_t q;
	queue_t *queue;
	acsm_pattern_t *mlist;
	acsm_pattern_t *px;

	mlist = 0;
	px = 0;
	queue = &q;

	/* init a queue */ 
	queue_init(queue);

	/* add the state 0 transitions 1st */ 
	for (i = 0; i < ALPHABET_SIZE; i++) {
		s = acsm->state_table[0].next_state[i];
		if (s) {
			queue_add(queue, s);
			acsm->state_table[s].fail_state = 0;
		}
	}

	/* build the fail state transitions for each valid state */ 
	while (queue_count(queue) > 0) {
		r = queue_remove(queue);

		/* find final states for any failure */ 
		for (i = 0; i < ALPHABET_SIZE; i++) {
			if ((s = acsm->state_table[r].next_state[i]) !=
			    ACSM_FAIL_STATE) {
				queue_add(queue, s);
				fs = acsm->state_table[r].fail_state;

				/* 
				 * locate the next valid state for 'i'
				 * starting at s 
				 */ 
				while ((next =
				    acsm->state_table[fs].next_state[i]) ==
				    ACSM_FAIL_STATE) {
					fs = acsm->state_table[fs].fail_state;
				}

				/*
				 * update 's' state failure state to point to
				 * the next valid state
				 */ 
				acsm->state_table[s].fail_state = next;

				/*
				 * copy 'next'states match_list to 's' states
				 * match_list, we copy them so each list can be 
				 * ac_free'd later, else we could just
				 * manipulate pointers to fake the copy.
				 */ 
				for (mlist = acsm->state_table[next].match_list; 
				    mlist != NULL ; mlist = mlist->next) {
					px = copy_match_list_entry(mlist);

					if (!px)
						printf("ERROR: no memory\n");

					/* insert at front of match_list */ 
					px->next =
					    acsm->state_table[s].match_list;
					acsm->state_table[s].match_list = px;
					acsm->state_table[s].num_finals++;
				}
			}
		}
	}

	/* clean up the queue */
	queue_free(queue);

	return;
}


/*
 * builds a Deterministic Finite Automaton from NFA
 */ 
static void
convert_NFA_to_DFA(acsm_t *acsm) 
{
	int i;
	int r;
	int s;
	queue_t q;
	queue_t *queue;

	queue = &q;

	/* init a queue */ 
	queue_init(queue);

	/* add the state 0 transitions 1st */ 
	for (i = 0; i < ALPHABET_SIZE; i++) {
		s = acsm->state_table[0].next_state[i];
		if (s)
			queue_add(queue, s);
	}

	/* start building the next layer of transitions */ 
	while (queue_count(queue) > 0) {
		r = queue_remove(queue);

		/* state is a branch state */ 
		for (i = 0; i < ALPHABET_SIZE; i++) {
			if ((s = acsm->state_table[r].next_state[i]) !=
			    ACSM_FAIL_STATE) {
				queue_add(queue, s);
			} else {
				acsm->state_table[r].next_state[i] =
				    acsm->state_table[acsm->state_table[r].
				    fail_state].next_state[i];
			}
		}
	}

	/* clean up the queue */ 
	queue_free(queue);

	return;
}


/* ================================== API =================================== */


/*
 * creates a new Aho-Corasick state machine
 */ 
acsm_t *
acsm_new(void) 
{
	acsm_t *p;

	init_xlatcase();
	p = (acsm_t *)ac_malloc(sizeof(acsm_t));
	MEMASSERT(p, "acsm_new");

	if (p)
		memset(p, 0, sizeof(acsm_t));

	return p;
}


/*
 * adds a pattern to the list of patterns for this state machine
 */ 
void
acsm_add_pattern(acsm_t *acsm, unsigned char *pat, int n, int nocase,
    int offset, int depth, void *id, int iid) 
{
	acsm_pattern_t *plist;

	plist = (acsm_pattern_t *)ac_malloc(sizeof(acsm_pattern_t));
	MEMASSERT(plist, "acsmAddPattern");
	plist->pattern = (unsigned char *)ac_malloc(n);
	MEMASSERT(plist->pattern, "acsmAddPattern");
	convert_case_ex(plist->pattern, pat, n);
	plist->casepattern = (unsigned char *)ac_malloc(n);
	MEMASSERT(plist->casepattern, "acsmAddPattern");
	memcpy(plist->casepattern, pat, n);

	plist->n	= n;
	plist->nocase	= nocase;
	plist->offset	= offset;
	plist->depth	= depth;
	plist->id	= id;
	plist->iid	= iid;
	plist->index	= acsm->num_patterns;
	plist->next	= acsm->patterns;

	acsm->patterns	= plist;

	acsm->num_patterns++;

	if (n > acsm->max_pattern_len)
		acsm->max_pattern_len = n;

	return;
}


/*
 * compiles the state machine
 */ 
void
acsm_compile(acsm_t *acsm) 
{
	int i;
	int k;
	acsm_pattern_t *plist;

	/* count number of states */ 
	acsm->max_states = 1;
	for (plist = acsm->patterns; plist != NULL; plist = plist->next)
		acsm->max_states += plist->n;
	acsm->state_table =
	    (acsm_state_table_t *)ac_malloc(sizeof(acsm_state_table_t) *
	    acsm->max_states);
	MEMASSERT(acsm->state_table, "Could not allocate state table");
	memset(acsm->state_table, 0, sizeof(acsm_state_table_t) *
	    acsm->max_states);

	/* initialize state zero as a branch */ 
	acsm->num_states = 0;

	/* initialize all states next_states to FAILED */ 
	for (k = 0; k < acsm->max_states; k++)
		for (i = 0; i < ALPHABET_SIZE; i++)
			acsm->state_table[k].next_state[i] = ACSM_FAIL_STATE;

	/* add each pattern to the state table */ 
	for (plist = acsm->patterns; plist != NULL; plist = plist->next)
		add_pattern_states(acsm, plist);

	/* set all failed state transitions to return to the 0'th state */ 
	for (i = 0; i < ALPHABET_SIZE; i++)
		if (acsm->state_table[0].next_state[i] == ACSM_FAIL_STATE)
			acsm->state_table[0].next_state[i] = 0;

	/* build the NFA  */ 
	build_NFA(acsm);

	/* convert the NFA to a DFA */ 
	convert_NFA_to_DFA(acsm);

	return;
}


/*
 * Creates the serialized DFA state table and transfers it to the device 
 */
void
acsm_gen_state_table(acsm_t *acsm, int mapped, cl_context ctx,
    cl_command_queue queue)
{
	FILE *fp;
	int i;
	int j;
	int k;
	int e;
	int state;
	int max_finals;
	int *finals;
	int temp_state[ALPHABET_SIZE];
	acsm_pattern_t *temp_ml;

	acsm->num_states = acsm->num_states + 1;

	/* allocate host and device memory for the serialized DFA */
	acsm->d_trans = clCreateBuffer(ctx,
	    CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)2 *
	    (size_t)ALPHABET_SIZE * (size_t)acsm->num_states * sizeof(cl_int),
	    NULL, &e);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: alloc d_trans: %s", clstrerror(e));

	if (mapped) {
		acsm->h_trans = clEnqueueMapBuffer(queue, acsm->d_trans,
		    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, (size_t)2 *
		    (size_t)ALPHABET_SIZE * (size_t)acsm->num_states *
		    sizeof(int), 0, NULL, NULL, &e);
		if (e != CL_SUCCESS)
			ERRXV(1, "ERROR: map d_tans: %s", clstrerror(e));
	} else {
		acsm->h_trans = MALLOC(acsm->num_states * 2 * ALPHABET_SIZE *
		      sizeof(int));
		if (!acsm->h_trans)
			 ERR(1, "ERROR: malloc h_trans");
	}

	/* loop through the states and write the state table to h_trans */
	for (i = 0; i < acsm->num_states; i++) {
		/* loop through the transitions */
		for (j = 0; j < ALPHABET_SIZE; j++) {
			state = acsm->state_table[i].next_state[j];
			/* final state */
			if (acsm->state_table[state].match_list) {
				acsm->h_trans[i * (2 * ALPHABET_SIZE) + j] =
				    -state;
				acsm->h_trans[i * (2 * ALPHABET_SIZE) +
				    ALPHABET_SIZE + j] = 
				    acsm->state_table[state].match_list->index;
			}
			/* normal state */
			else {
				acsm->h_trans[i * (2 * ALPHABET_SIZE) + j] =
				    state;
			}
		}
	}
	acsm->size = acsm->num_states * (ALPHABET_SIZE * 2) * sizeof(int);

	if (mapped)
		return;

	e = clEnqueueWriteBuffer(queue, acsm->d_trans, CL_TRUE, 0, (size_t)2 *
	    (size_t)ALPHABET_SIZE * (size_t)acsm->num_states * sizeof(cl_int),
	    acsm->h_trans, 0, NULL, NULL);
	if (e != CL_SUCCESS)
		ERRXV(1, "ERROR: write d_trans: %s", clstrerror(e));

	return;
}

/*
 * returns a newly allocated table containing all patterns contained
 * in this acsm_t
 */
acsm_pattern_t *
acsm_get_patterns_table(acsm_t *acsm)
{
	int i;
	acsm_pattern_t *p = NULL, *q = NULL;

	if (acsm == NULL) {
		return NULL;
	}

	acsm_pattern_t *patterns =
		MALLOC(acsm->num_patterns * sizeof(acsm_pattern_t));

	p = acsm->patterns;
	while (p) {
		int x = p->index;
		patterns[x].pattern     = strndup(p->pattern, p->n);
		patterns[x].casepattern = strndup(p->casepattern, p->n);
		patterns[x].n           = p->n;
		patterns[x].nocase      = p->nocase;
		patterns[x].offset      = p->offset;
		patterns[x].depth       = p->depth;
		patterns[x].id          = p->id;
		patterns[x].iid         = p->iid;
		patterns[x].index       = p->index;
		patterns[x].next        = NULL;

		p = p->next;
	}

	for (i = 0; i < acsm->max_states; i++) {
		p = acsm->state_table[i].match_list;
		if (p && (p->next)) {
			q = &patterns[p->index];

			while (q->next)
				q = q->next;

			while (p && (p->next)) {
				q->next = &patterns[p->next->index];
				p = p->next;
				q = q->next;
			}
		}
	}

	/* print patterns and connections between them (i.e., common prefix) */
	//for (i=0; i<acsm->num_patterns; i++) {
	//	printf("%d %d %s", patterns[i].index, patterns[i].iid, patterns[i].pattern);
	//	p = patterns[i].next;
	//	while (p) {
	//		printf(" -> %d %s", p->index, p->pattern);
	//		p = p->next;
	//	}
	//	printf("\n");
	//}

	return patterns;
}


/*
 * returns the size of the largest pattern
 */
int
acsm_get_max_pattern_size(acsm_t *acsm)
{
	return acsm->max_pattern_len;
}


/*
 *returns the number of states in the automaton
 */
int
acsm_get_states(acsm_t *acsm)
{
	return acsm->num_states;
}


/*
 * returns the automaton size in bytes
 */
size_t
acsm_get_size(acsm_t *acsm)
{
	return acsm->size;
}


/*
 * cleans the memory and keeps the serialized DFA state table
 */ 
void
acsm_cleanup(acsm_t *acsm) 
{
	int i;
	acsm_pattern_t *mlist, *ilist;

	for (i = 0; i < acsm->max_states; i++) {
		mlist = acsm->state_table[i].match_list;
		while (mlist) {
			ilist = mlist;
			mlist = mlist->next;
			ac_free(ilist);
		}
		acsm->state_table[i].match_list = NULL;
	}

	ac_free(acsm->state_table);
	acsm->state_table = NULL;

	mlist = acsm->patterns;

	while (mlist) {
		ilist = mlist;
		mlist = mlist->next;
		ac_free(ilist->pattern);
		ac_free(ilist->casepattern);
		ac_free(ilist);
	}

	acsm->patterns = NULL;

	return;
}


/*
 * frees all acsm memory
 */ 
void
acsm_free(acsm_t *acsm) 
{
	ac_free(acsm);

	return;
}
