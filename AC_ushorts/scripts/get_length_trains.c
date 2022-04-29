/*
 * author: epapado
 * a script that:
 * parses an ascii flow trace with name 
 * ``srcip-dstip-srcport-dstport'' (generated via tshark)
 * and prints the packet payload trains in three different
 * files to indicate the destination of each packet.
 * packet lenghts with negative value indicate that these 
 * packets were sent from the server.  
 */

#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <dirent.h> 
  
int 
main(void) 
{ 
	char line[1024];
	char *token, *lentoken, *len, *file, *tmp;
	char *iptoken, *srcip, *dstip, *this_srcip, *this_dstip;
	struct dirent *d;  
	int tid;
	char *tx = "tx.signatures";
	char *rx = "rx.signatures";
	char *txrx = "txrx.signatures";
	
	FILE *fd, *txfd, *rxfd, *txrxfd;
	DIR *dr = opendir("res/exploits.out/5.pcap/"); 

	if (dr == NULL) { 
		printf("Could not open current directory" ); 
		exit(EXIT_FAILURE); 
	} 
	
	txfd = fopen(tx, "a");
	rxfd = fopen(rx, "a");
	txrxfd = fopen(txrx, "a");
	if (txfd == NULL || rxfd == NULL || txrxfd == NULL) {
		printf("fopen error\n");
		exit(EXIT_FAILURE); 
	}

	while ((d = readdir(dr)) != NULL) {
		printf("%s\n", d->d_name);
		if (strcmp(d->d_name, ".") == 0 || strcmp(d->d_name, "..") == 0)
			continue;
		tmp = strdup(d->d_name); 
		iptoken = strtok(tmp, "-");
		srcip = strdup(iptoken);
		iptoken = strtok(NULL, "-");
		dstip = strdup(iptoken);

		file = malloc(512);
		if (file == NULL) {
			printf("malloc error\n");
			exit(EXIT_FAILURE); 
		}
		file = strcpy(file, "res/exploits.out/5.pcap/");
		file = strcat(file, d->d_name);
		//file = strcat(file, d->d_name);
		fd = fopen(file, "r");
		if (fd == NULL) {
			printf("fopen error\n");
			exit(EXIT_FAILURE); 
		}
		while(fgets(line, 1024, fd) != NULL) {
			line[strlen(line) - 1] = '\0';
			token = strtok(line, " ");
			tid = 1;
			while (token != NULL) {
				if (tid == 3)
					this_srcip = strdup(token);
				if (tid == 5)
					this_dstip = strdup(token);
				if (strstr(token, "Len=") != NULL) {
					tmp = strdup(token);
					lentoken = strtok(tmp, "=");
					lentoken = strtok(NULL, "=");
					len = strdup(lentoken);
					break;
				}
				token = strtok(NULL, " ");
				tid++;
			}
			if (strcmp(this_srcip, srcip) == 0) {
				printf("%s, ", len);
				fprintf(txfd, "%s, ", len);
				fprintf(txrxfd, "%s, ", len);
			} else {
				printf("-%s, ", len);
				fprintf(rxfd, "%s, ", len);
				fprintf(txrxfd, "-%s, ", len);
			}
		}
		printf("\n");
		fprintf(txfd, "\n");
		fprintf(rxfd, "\n");
		fprintf(txrxfd, "\n");

		free(file);
		fclose(fd);
	}
	fclose(txfd);
	fclose(rxfd);
	fclose(txrxfd);
	closedir(dr);     
	return 0; 
}
