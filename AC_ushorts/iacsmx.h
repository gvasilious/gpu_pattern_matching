/*
 * Copyright (C) 2002 Martin Roesch <roesch@sourcefire.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 2 as
 * published by the Free Software Foundation.  You may not use, modify or
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
 *  author:	Dimitris Deyannis
 *  contact:	deyannis@ics.forth.gr
 *  changes by author:
 *	Sep 2016: changed the (horrible whatever it was) style to OpenBSD
 *	      style(9)
 *	Oct 2016: major refactoring, added dump to .bin, added iid table dump,
 *	          removed dead code
 *	Nov 2016: changed functions returning just zero to void, changed
 *	          declaring and initializing variables at the same time,
 *	          partially documented the code with comments for the newcomers
 *	Mar 2017: added transition table to OpenCL buffers, removed dumping to
 *	          file for current version, removed iid table dump
 */

#ifndef _IACSMX_H_
#define _IACSMX_H_


#include <stdio.h>
#include <stdlib.h>

#include <CL/opencl.h>


#define I_ALPHABET_SIZE	2048
//#define I_ALPHABET_SIZE	16384

/* default Fail State */
#define ACSM_FAIL_STATE	-1


/* Aho-Corasick state machine pattern */
struct _iacsm_pattern {
	struct  _iacsm_pattern	*next;
	unsigned short	*pattern;
	int			n;
	int			offset;
	int			depth;
	void			*id;
	int			iid;
};
typedef struct _iacsm_pattern iacsm_pattern_t;


/* Aho-Corasick state machine state table */
struct _iacsm_state_table {
	int		next_state[I_ALPHABET_SIZE];
	int		fail_state;
	int		num_finals;
	iacsm_pattern_t	*match_list;
};
typedef struct _iacsm_state_table iacsm_state_table_t;


/* Aho-Corasick state machine */
struct _iacsm {
	int			max_states;
	int			num_states;
	int			max_pattern_len;
	size_t			size;
	iacsm_pattern_t		*patterns;
	iacsm_state_table_t	*state_table;
	int			*h_trans;
	cl_mem			d_trans;
};
typedef struct _iacsm iacsm_t;


/*
 * creates a new Aho-Corasick state machine
 *
 * ret: a new Aho-Corasick state machine
 */
iacsm_t *
iacsm_new(void);


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
iacsm_add_pattern(iacsm_t *, unsigned short *, int, int, int, void *, int);


void
iacsm_add_fullpattern(iacsm_t *, const char *, int);


/*
 * compiles the state machine
 *
 * arg0: Aho-Corasick state machine
 */
void
iacsm_compile(iacsm_t *);


/*
 * creates the serialized DFA state table and transfers it to the device
 *
 * arg0: Aho-Corasick state machine
 * arg1: memory mapped buffers flag
 * arg2: OpenCL context
 * arg3: OpenCL command queue
 */
void
iacsm_gen_state_table(iacsm_t *, int, cl_context, cl_command_queue);


/*
 * returns the size of the largest pattern
 *
 * arg0: Aho-Corasick state machine
 *
 * ret:  size of the largest pattern in bytes
 */
int
iacsm_get_max_pattern_size(iacsm_t *);


/*
 * returns the number of states in the automaton
 *
 * arg0: Aho-Corasick state machine
 *
 * ret:  number of states
 */
int
iacsm_get_states(iacsm_t *);


/*
 * returns the automaton size in bytes
 *
 * arg0: Aho-Corasick state machine
 *
 * ret:  automaton size in bytes
 */
size_t
iacsm_get_size(iacsm_t *);


/*
 * cleans the memory and keeps the serialized DFA state table
 *
 * arg0: Aho-Corasick state machine
 */
void
iacsm_cleanup(iacsm_t *);


/*
 * frees all acsm memory
 *
 * arg0: Aho-Corasick state machine
 */
void
iacsm_free(iacsm_t *);


#endif /* _IACSMX_H_ */
