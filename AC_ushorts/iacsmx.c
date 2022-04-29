#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <CL/opencl.h>

#include "iacsmx.h"
#include "common.h"
#include "clutil.h"

#define MAX_PATT_LEN 16
#define IACSM_FAIL_STATE -1
#define MEMASSERT(p, s) do {                                            \
        if (!p) {                                                       \
                fprintf(stderr, "ERROR! out off memory: %s!\n", s);     \
                exit(0);                                                \
        }                                                               \
} while (0)

/* ==== queue ==== */

/*
 * simple queue_t node
 */
struct _qnode {
        int             state;
        struct _qnode   *next;
};
typedef struct _qnode qnode_t;


/*
 * simple queue
 */
struct _queue {
        qnode_t *head;
        qnode_t *tail;
        int     count;
};
typedef struct _queue queue_t;


/* ==== memory handling wrapping ==== */

static void *
iac_malloc(size_t n) 
{
	void *p;

	p = calloc(1, n);

	return p;
}


/*
 * free wrapper
 */
static void
iac_free(void *p) {
        if (p)
                free (p);

        return;
}


/* ==== queue ==== */

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
                q = s->tail = s->head = iac_malloc(sizeof(qnode_t));
                MEMASSERT(q, "queue_add");
                q->state = state;
                q->next = 0;
        } else  {
                q = iac_malloc(sizeof(qnode_t));
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
                iac_free(q);
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

/*===============================*/

iacsm_t *
iacsm_new(void) {

	iacsm_t *p;

	p = iac_malloc(sizeof(iacsm_t));
	MEMASSERT(p, "iacsm_new");

	if (p)
		memset(p, 0, sizeof(iacsm_t));

	return p;
}

/*
 * copies a match list entry
 */
static iacsm_pattern_t *
copy_match_list_entry(iacsm_pattern_t *px)
{
        iacsm_pattern_t *p;

        p = iac_malloc(sizeof(iacsm_pattern_t));
        MEMASSERT(p, "copy_match_list_entry");
        memcpy(p, px, sizeof(iacsm_pattern_t));
        p->next = 0;

        return p;
}

static void
build_NFA(iacsm_t *iacsm)
{
	int i, r, s, fs, next;
	queue_t q;
	queue_t *queue;
	iacsm_pattern_t *mlist;
	iacsm_pattern_t *px;

	mlist = 0;
	px = 0;
	queue = &q;


        /* init a queue */
        queue_init(queue);

        /* add the state 0 transitions 1st */
        for (i = 0; i < I_ALPHABET_SIZE; i++) {
                s = iacsm->state_table[0].next_state[i];
                if (s) {
                        queue_add(queue, s);
                        iacsm->state_table[s].fail_state = 0;
                }
        }
	
	/* build the fail state transitions for each valid state */
        while (queue_count(queue) > 0) {
                r = queue_remove(queue);

                /* find final states for any failure */
                for (i = 0; i < I_ALPHABET_SIZE; i++) {
                        if ((s = iacsm->state_table[r].next_state[i]) != IACSM_FAIL_STATE) {
                                queue_add(queue, s);
                                fs = iacsm->state_table[r].fail_state;
        
                                /*
                                 * locate the next valid state for 'i'
                                 * starting at s
                                 */
                                while ((next = iacsm->state_table[fs].next_state[i]) == IACSM_FAIL_STATE) {
                                        fs = iacsm->state_table[fs].fail_state;
                                }
        
                                /*
                                 * update 's' state failure state to point to
                                 * the next valid state
                                 */
                                iacsm->state_table[s].fail_state = next;
        
                                /*
                                 * copy 'next'states match_list to 's' states
                                 * match_list, we copy them so each list can be
                                 * ac_free'd later, else we could just
                                 * manipulate pointers to fake the copy.
                                 */
                                for (mlist = iacsm->state_table[next].match_list; mlist != NULL ; mlist = mlist->next) {
                                        px = copy_match_list_entry(mlist);
        
                                        if (!px)
                                                printf("ERROR: no memory\n");
        
                                        /* insert at front of match_list */
                                        px->next = iacsm->state_table[s].match_list;
                                        iacsm->state_table[s].match_list = px;
                                        iacsm->state_table[s].num_finals++;
                                }
                        }
                }
	}

	queue_free(queue);

	return;
}


/*
 * builds a Deterministic Finite Automaton from NFA
 */
static void
convert_NFA_to_DFA(iacsm_t *iacsm)
{
        int i, r, s;
        queue_t q;
        queue_t *queue;

        queue = &q;

        /* init a queue */
        queue_init(queue);

        /* add the state 0 transitions 1st */
        for (i = 0; i < I_ALPHABET_SIZE; i++) {
                s = iacsm->state_table[0].next_state[i];
                if (s)
                        queue_add(queue, s);
        }

        /* start building the next layer of transitions */
        while (queue_count(queue) > 0) {
                r = queue_remove(queue);

                /* state is a branch state */
                for (i = 0; i < I_ALPHABET_SIZE; i++) {
                        if ((s = iacsm->state_table[r].next_state[i]) != IACSM_FAIL_STATE) {
                                queue_add(queue, s);
                        } else {
                                iacsm->state_table[r].next_state[i] = 
					iacsm->state_table[iacsm->state_table[r].fail_state].next_state[i];
                        }
                }
        }

        /* clean up the queue */
        queue_free(queue);

        return;
}

static void
add_match_list_entry(iacsm_t *iacsm, int state, iacsm_pattern_t *px)
{
	iacsm_pattern_t *p;

	p = iac_malloc(sizeof(iacsm_pattern_t));
	MEMASSERT(p, "add_match_list_entry");
	memcpy(p, px, sizeof(iacsm_pattern_t));

	p->next = iacsm->state_table[state].match_list;
	iacsm->state_table[state].match_list = p;
	iacsm->state_table[state].num_finals++;

	return;
}


static void
add_pattern_states(iacsm_t *iacsm, iacsm_pattern_t *p)
{
	unsigned short *pattern;
	int state; 
	int next;
	int len;

	len = p->n;
	pattern = p->pattern;
	state = 0;

	for (; len > 0; pattern++, len--) {
		next = iacsm->state_table[state].next_state[*pattern];
		if (next == IACSM_FAIL_STATE)
			break;
		state = next;
	}

	for (; len > 0; pattern++, len--) {
		iacsm->num_states++;
		iacsm->state_table[state].next_state[*pattern] = iacsm->num_states;
		state = iacsm->num_states;
	}

	add_match_list_entry(iacsm, state, p);

	return;
}

	
void
iacsm_compile(iacsm_t *iacsm)
{
	int i, j;
	iacsm_pattern_t *plist;

	iacsm->max_states = 1;
	for (plist = iacsm->patterns; plist != NULL; plist = plist->next)
		iacsm->max_states += plist->n;
	iacsm->state_table = iac_malloc(sizeof(iacsm_state_table_t) * iacsm->max_states);
	MEMASSERT(iacsm->state_table, "iacsm_compile");
	memset(iacsm->state_table, 0, sizeof(iacsm_state_table_t) * iacsm->max_states);

	iacsm->num_states = 0;

	for (i = 0; i < iacsm->max_states; i++)
		for (j = 0; j < I_ALPHABET_SIZE; j++)
			iacsm->state_table[i].next_state[j] = IACSM_FAIL_STATE;

	for (plist = iacsm->patterns; plist != NULL; plist = plist->next)
		add_pattern_states(iacsm, plist);

	for (i = 0; i < I_ALPHABET_SIZE; i++)
		if (iacsm->state_table[0].next_state[i] == IACSM_FAIL_STATE)
			iacsm->state_table[0].next_state[i] = 0;

	build_NFA(iacsm);
	convert_NFA_to_DFA(iacsm);

	return;
}
	

void 
iacsm_add_pattern(iacsm_t *iacsm, unsigned short *items, int len, 
		int offset, int depth, void *id, int iid)
{
	int i;
	iacsm_pattern_t *plist;

	plist = iac_malloc(sizeof(iacsm_pattern_t));
	MEMASSERT(plist, "iacsm_add_pattern");
	plist->pattern = iac_malloc(sizeof(len));
	MEMASSERT(plist->pattern, "iacsm_add_pattern");
	memcpy(plist->pattern, items, len * sizeof(unsigned short)); 

	plist->n        = len;
	plist->offset   = offset;
	plist->depth    = depth;
	plist->id       = id;
	plist->iid      = iid;
	plist->next     = iacsm->patterns;
	iacsm->patterns  = plist;
	if (len > iacsm->max_pattern_len)
        	iacsm->max_pattern_len = len;

	return;

}


void
iacsm_add_fullpattern(iacsm_t *iacsm, const char *pattern, int np)
{
        int i, j, len;
	char *subpattern;
	unsigned short item;
	unsigned short items[MAX_PATT_LEN];


	subpattern = malloc(strlen(pattern) * sizeof(char));
	if (subpattern == NULL)
		ERRX(1, "malloc");

	i = 0;
	j = 0;
	len = 0;
	/* break patterns into subpatterns with the comma character as delimiter */
	while (i < strlen(pattern) + 1) {
        	if ((pattern[i] == ',') || (pattern[i] == '\n') || 
				(pattern[i] == '\0') || (pattern[i] == '\r')) {
		      subpattern[j] = '\0';
		      /* then, convert into unsigned short */
		      item = atoi(subpattern); 
		      items[len++] = item; 
		      j = 0;
		}
		else {
			subpattern[j] = pattern[i];
			j++;
		}
		i++;	
	}
	iacsm_add_pattern(iacsm, items, len, 0, 0 , 0, np);
	return;
}


void
iacsm_gen_state_table(iacsm_t *iacsm, int mapped, cl_context ctx,
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
        int temp_state[I_ALPHABET_SIZE];
        iacsm_pattern_t *temp_ml;

        iacsm->num_states = iacsm->num_states + 1;

        /* allocate host and device memory for the serialized DFA */
        iacsm->d_trans = clCreateBuffer(ctx,
            CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, (size_t)2 *
            (size_t)I_ALPHABET_SIZE * (size_t)iacsm->num_states * sizeof(cl_int),
            NULL, &e);
        if (e != CL_SUCCESS)
                ERRXV(1, "ERROR: alloc d_trans: %s", clstrerror(e));

        if (mapped) {
                iacsm->h_trans = clEnqueueMapBuffer(queue, iacsm->d_trans,
                    CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, (size_t)2 *
                    (size_t)I_ALPHABET_SIZE * (size_t)iacsm->num_states *
                    sizeof(int), 0, NULL, NULL, &e);
                if (e != CL_SUCCESS)
                        ERRXV(1, "ERROR: map d_tans: %s", clstrerror(e));
        } else {
                iacsm->h_trans = MALLOC(iacsm->num_states * 2 * I_ALPHABET_SIZE *
                      sizeof(int));
                if (!iacsm->h_trans)
                         ERR(1, "ERROR: malloc h_trans");
        }

        /* loop through the states and write the state table to h_trans */
        for (i = 0; i < iacsm->num_states; i++) {
                /* loop through the transitions */
                for (j = 0; j < I_ALPHABET_SIZE; j++) {
			state = iacsm->state_table[i].next_state[j];
                        /* final state */
                        if (iacsm->state_table[state].match_list) {
                                iacsm->h_trans[i * (2 * I_ALPHABET_SIZE) + j] =
                                    -state;
                                iacsm->h_trans[i * (2 * I_ALPHABET_SIZE) +
                                    I_ALPHABET_SIZE + j] =
                                    iacsm->state_table[state].match_list->iid;
                        }
                        /* normal state */
                        else {
                                iacsm->h_trans[i * (2 * I_ALPHABET_SIZE) + j] =
                                    state;
                        }
                }
        }
        iacsm->size = iacsm->num_states * (I_ALPHABET_SIZE * 2) * sizeof(int);

        if (mapped)
                return;

        e = clEnqueueWriteBuffer(queue, iacsm->d_trans, CL_TRUE, 0, (size_t)2 *
            (size_t)I_ALPHABET_SIZE * (size_t)iacsm->num_states * sizeof(cl_int),
            iacsm->h_trans, 0, NULL, NULL);
        if (e != CL_SUCCESS)
                ERRXV(1, "ERROR: write d_trans: %s", clstrerror(e));

        return;
}


/*
 * returns the size of the largest pattern
 */
int
iacsm_get_max_pattern_size(iacsm_t *iacsm)
{
        return iacsm->max_pattern_len;
}


/*
 *returns the number of states in the automaton
 */
int
iacsm_get_states(iacsm_t *iacsm)
{
        return iacsm->num_states;
}


/* XXX
 * returns the automaton size in bytes
 */
size_t
iacsm_get_size(iacsm_t *iacsm)
{
        return iacsm->size;
}


/*
 * cleans the memory and keeps the serialized DFA state table
 */
void
iacsm_cleanup(iacsm_t *iacsm)
{
        int i;
        iacsm_pattern_t *mlist, *ilist;

        for (i = 0; i < iacsm->max_states; i++) {
                mlist = iacsm->state_table[i].match_list;
                while (mlist) {
                        ilist = mlist;
                        mlist = mlist->next;
                        iac_free(ilist);
                }
        }

        iac_free(iacsm->state_table);
        mlist = iacsm->patterns;

        while (mlist) {
                ilist = mlist;
                mlist = mlist->next;
                iac_free(ilist->pattern);
                iac_free(ilist);
        }

        return;
}


/*
 * frees all acsm memory
 */
void
iacsm_free(iacsm_t *iacsm)
{
        iac_free(iacsm);

        return;
}

//void
//usage()
//{
//	printf("usage \n");
//
//	return;
//}


//int main(int argc, char *argv[])
//{
//
//	int opt, np;
//	char *in;
//	char line[1000];
//	FILE *fp;
//
//	iacsm_t *iacsm;
//
//	while ((opt = getopt(argc, argv, "i:h")) != -1) {
//		switch (opt) {
//			case 'i':
//				in = strdup(optarg);
//				break;
//			case 'h':
//			default:
//				usage();
//		}
//	}
//
//	fp = fopen(in, "r");
//	if (fp == NULL)
//		ERRXV(1, "fopen: %s", in);
//       
//	iacsm = iacsm_new();
//
//	np = 0;
//	while (fgets(line, sizeof(line), fp)) {
//		if (line[strlen(line) - 1] == '\n')
//			line[strlen(line) - 1] = '\0'; /* strip newline */
//		printf("%d: %s\n", np, line);
//		iacsm_add_fullpattern(iacsm, line, np);
//		np++;
//	}
//	fclose(fp);
//
//	iacsm_compile(iacsm);
//	iacsm_gen_state_table(iacsm);
//	iacsm_cleanup(iacsm);
//
//	return 0;
//}

