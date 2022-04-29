/*
 * author:	Dimitris Deyannis
 * last update:	Mar-2017
 * contact:	deyannis@ics.forth.gr
 */

#include <unistd.h>
#include <time.h>
#include <ctype.h>

#include "utils.h"
#include "common.h"


/*
 * converts half printable hex value to integer
 */
int
half_hex_to_int(unsigned char c)
{
	if (isdigit(c))
		return c - '0';

	if ((tolower(c) >= 'a') && (tolower(c) <= 'f'))
		return tolower(c) + 10 - 'a';
}


/*
 * converts a printable hex array to ASCII
 */
unsigned char *
printable_hex_to_bytes(unsigned char *input)
{
	int i;
	unsigned char *output;

	output = NULL;
	if (strlen(input) % 2 != 0) {
		printf("ERROR: reading pattern!\n");
		exit(EXIT_FAILURE);
	}

	output = calloc(strlen(input) / 2, sizeof(unsigned char));
	if (!output)
		ERRX(1, "ERROR: calloc output\n");

	for (i = 0; i < strlen(input); i+= 2) {
		output[i / 2] = ((unsigned char)half_hex_to_int(input[i])) *
		    16 + ((unsigned char)half_hex_to_int(input[i + 1]));
	}
	
	return output;
}


/*
 * returns the current time in usecs
 */
size_t
gettime(void)
{
	struct timespec tp;

	clock_gettime(CLOCK_MONOTONIC, &tp);

	return (tp.tv_sec * 1000000 + tp.tv_nsec / 1000.0);
}
