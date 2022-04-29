#include <dirent.h> 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <linux/limits.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "file_traverse.h"
#include "common.h"


/*
 * checks if given file exists.
 */
int
file_exists(char *filename)
{
	struct stat buffer;

	if (filename == NULL)
		return 0;

	return (stat (filename, &buffer) == 0);
}

/*
 * checks if the path points to a regular file
 */
int
is_regular_file(const char *path)
{
	struct stat path_stat;
		
	stat(path, &path_stat);
		
	return S_ISREG(path_stat.st_mode);
}


/*
 * checks if the path points to a directory
 */
int
is_directory(const char *path)
{
	struct stat path_stat;

	stat(path, &path_stat);

	return S_ISDIR(path_stat.st_mode);
}


#ifdef PRINT_ALL
/*
 * prints everything under path
 */
static void
print_all(char *path)
{
	char d_path[PATH_MAX + NAME_MAX];
	DIR *d;
	struct dirent *dir;

	d = opendir(path);
	if (d == NULL)
		return;

	while ((dir = readdir(d)) != NULL) {
		if (dir-> d_type != DT_DIR)
      			printf("%s/%s\n", path, dir->d_name);
		else {
			if ((dir->d_type == DT_DIR)         && 
			    (strcmp(dir->d_name, ".") != 0) && 
			    (strcmp(dir->d_name, "..") != 0)) {
				printf("%s/%s\n", path, dir->d_name);

				sprintf(d_path, "%s/%s", path, dir->d_name);

				print_all(d_path);
			}
		}
	}

	closedir(d);
}
#endif /* PRINT_ALL */


/*
 * returns a newly allocated string containing the names
 * of all the regular files under the requested directory
 */
char *
get_all_regular_files(char *path)
{
	int files_len;
	int files_sz;
	int len;
	int path_len;
	char file[PATH_MAX + NAME_MAX];
	char *files;
	DIR *d;
	struct dirent *dir;

	files_len = 0;
	files_sz  = 0;
	len       = 0;

	if (path == NULL)
		return NULL;

	path_len = strlen(path);

	if (path[path_len - 1] == '/') {
		path[path_len - 1] = '\0';
		path_len--;
	}

	files_sz = PATH_MAX + NAME_MAX;
	files = calloc(files_sz, sizeof(char));
	if (!files)
		return NULL;

	d = opendir(path);
	if (d == NULL)
		return NULL;

	while ((dir = readdir(d)) != NULL) {
		if (dir-> d_type != DT_DIR) {
      		    len = sprintf(file, "%s/%s", path, dir->d_name);

			if (is_regular_file(file)) {
				if (files_len + len >= files_sz) {
					files_sz += PATH_MAX + NAME_MAX;
					files = realloc(files, files_sz);
					if (!files)
						return NULL;
				}
				strcat(files, ",");
				strcat(files, file);
				files_len += len + 1;
			}
		}
	}

	closedir(d);

	/* remove trailing , */
	files[files_len] = '\0';

	return files;
}



#ifdef TEST_MAIN
int
main(int argc, char **argv)
{
	char *f;
#if 0
	printf("%s\n", NORMAL_COLOR);

	show_dir_content(argv[1]);

	printf("%s\n", NORMAL_COLOR);
#endif
	printf("Testing 'get_all_regular_files'\n");

	f = get_all_regular_files(argv[1]);
	if (!f)
		printf("Failed\n");
	else
		printf("%s\n", f);

	free(f);

	return 0;
}
#endif /* TEST_MAIN */
