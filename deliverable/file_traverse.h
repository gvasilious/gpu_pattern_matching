#ifndef _FILE_TRAVERSE_H_
#define _FILE_TRAVERSE_H_


/* enables the print_all function */
/* #define PRINT_ALL */


/*
 * checks if given file exists.
 *
 * arg0: path to input file
 *
 * ret:  0 if the file in the path does not exist
 *       1 if the file in the path exists
 */
int
file_exists(char *);


/*
 * checks if given file is regular
 *
 * arg0: path to input file
 *
 * ret:  0 if the file in the path is not regular
 *       1 if the file in the path is regular
 */
int
is_regular_file(const char *);


/*
 * checks if the path points to a directory
 *
 * arg0: path
 *
 * ret:  0 if the path does not point to a directory
 *       1 if the path points to a directory
 */
int
is_directory(const char *);


/*
 * returns a newly allocated string containing the names
 * of all the regular files under the requested directory
 *
 * arg0: path to directory
 *
 * ret:  a string containing all the file names in the directory,
 *       comma separated
 */
char *
get_all_regular_files(char *);


#endif /* _FILE_TRAVERSE_H_ */
