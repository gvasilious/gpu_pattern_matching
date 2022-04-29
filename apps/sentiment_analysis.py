import subprocess
import shlex
import sys
import signal
import os
import math
import time
import datetime

# a class that implements time-decaying counters using an
# exponential formula of the form c = c * e^(lamda)*Dt
# where lamda = ln(2) / halflife.
# halflife is the required time window.
class TimeWindowCounter:
    def __init__(self, halflife):
        self.halflife = halflife
        self.counter = 0;
        self.timestamp = 0;

    def inc(self, timestamp_new):
        decayrate = math.log(2) / self.halflife
        if self.timestamp == 0:
            self.timestamp = timestamp_new

        dt = timestamp_new - self.timestamp

        self.counter = 1 + math.exp(-decayrate * dt) * self.counter
        self.timestamp = timestamp_new

    def inc(self, value, timestamp_new):
        decayrate = math.log(2) / self.halflife
        if self.timestamp == 0:
            self.timestamp = timestamp_new

        dt = timestamp_new - self.timestamp

        self.counter = value + math.exp(-decayrate * dt) * self.counter
        self.timestamp = timestamp_new

    def update(self, timestamp_new):
        decayrate = math.log(2) / self.halflife
        if self.timestamp == 0:
            self.timestamp = timestamp_new

        dt = timestamp_new - self.timestamp

        self.counter = math.exp(-decayrate * dt) * self.counter
        self.timestamp = timestamp_new
        return self.counter

    def get(self):
            return self.counter


patternsIdTable       = {} # the patterns set
patternsTable         = {} # the IDs and the patterns
patternsMetadataTable = {} # patterns metadata (if any)

negativePatternId = 0
positivePatternId = 0

USE_POSITIVE_WORDS_LIST = True
USE_NEGATIVE_WORDS_LIST = True

# both negative and positive words will be stored in the same file
fout = open("patterns.txt", "w")

if USE_NEGATIVE_WORDS_LIST == True:
    # load negative words and assign them with a unique negative ID
    with open("patterns/sentiment/negative_words_en.txt", "r") as fin:
        for line in fin:
            patrn = line.strip('\n')
            negativePatternId -= 1
            patternsIdTable[patrn] = negativePatternId
            patternsTable[negativePatternId] = patrn
            fout.write(str(negativePatternId) + " "  + "\" " + patrn + " \"" + "\n")
        fin.close()


if USE_POSITIVE_WORDS_LIST == True:
    # load positive words, assign them with a unique positive ID
    positivePatternId=0
    with open("patterns/sentiment/positive_words_en.txt", "r") as fin:
        for line in fin:
            patrn = line.strip('\n')
            positivePatternId += 1
            patternsIdTable[patrn] = positivePatternId
            patternsTable[positivePatternId] = patrn
            fout.write(str(positivePatternId) + " " +  "\" " + patrn + " \"" + "\n")
        fin.close()

# load top-5000 words of the 2000s, together with their scores
pat_id=0
with open("patterns/sentiment/top-5000_2000decade.txt", "r") as fin:
    for line in fin:
        line = line.strip('\n')
        lineList = line.split()

        if len(lineList) != 3:
            print("Bad format: ", lineList)
            continue

        patrn = lineList[0]
        patrn_score_mean = float(lineList[1])
        patrn_score_std  = float(lineList[2])

        if patrn in patternsIdTable:
            pat_id = patternsIdTable[patrn]
            patternsMetadataTable[pat_id] = patrn_score_mean
        else:
            if patrn_score_mean < 0:
                negativePatternId -= 1
                pat_id = negativePatternId
            else:
                positivePatternId += 1
                pat_id = positivePatternId

            patternsIdTable[patrn] = pat_id
            patternsTable[pat_id] = patrn
            patternsMetadataTable[pat_id] = [patrn_score_mean, patrn_score_std]
            fout.write(str(pat_id) + " " +  "\" " + patrn + " \"" + "\n")


    fin.close()

# all patterns have been stored
fout.close()

##############################################################################
##############################################################################

timeWindows = [60, 3600, 3600*8, 3600*24, 3600*24*7]

positiveCountersTable = {}
negativeCountersTable = {}
wordsFrequenciesTable = {}

for i in timeWindows:
    positiveCountersTable.update({i : TimeWindowCounter(i)})
    negativeCountersTable.update({i : TimeWindowCounter(i)})
    wordsFrequenciesTable.update({i : {}})

nlines = 0;
nmatches = 0;

def handler(signum, frame):

    currentTime = time.time()

    for i in timeWindows:
        print(datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"), round(currentTime, 1), str(i).rjust(8), end=" : ")

        cnt_pos = positiveCountersTable[i]
        cnt_neg = negativeCountersTable[i]
        tab_freqs = wordsFrequenciesTable[i]

        cnt_pos.update(currentTime)
        cnt_neg.update(currentTime)

        if cnt_pos.get() > 0 or cnt_neg.get() > 0:
            score = cnt_pos.get() / (cnt_pos.get() + cnt_neg.get())
            print("Score: ", round(score * 100, 1), "%", end=" ")

            if len(tab_freqs) > 0:
                # update and sort frequencies to get the heavy-hitters
                l = sorted(tab_freqs.items(), key = lambda kv:(kv[1].update(currentTime), kv[0]), reverse=True)

                print("--------", end="")

                #print Top-K words
                print("[", end=" ")
                loops = min(5, len(l))
                for i in range(0, loops):
                    pat_id = int(l[i][0])
                    pat_cnt = l[i][1].get()
                    print(patternsTable[pat_id].rjust(10), "(", round(pat_cnt, 1), ")", end=" ")

                print("]")
        else:
            print("")
            
    signal.alarm(5)

# Set the signal handler and a 5-second alarm
signal.signal(signal.SIGALRM, handler)
signal.alarm(1)

command="./ocl_aho_grep -p patterns.txt -f " + sys.argv[1] + "  -B 4096 -D 0 -L 1024 -G 8192 -w 1  -v " + sys.argv[2]
process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)

while process.poll() is None:
    output = process.stdout.readline()

    output = str(output, 'utf-8')
    if output.find('Pattern') == 0:

        pid = output.split()[1]
        pid = pid.replace('#', '')

        currentTime = time.time()

        for i in timeWindows:
            cn = negativeCountersTable[i]
            cp = positiveCountersTable[i]
            wf = wordsFrequenciesTable[i]

            score = 0
            if pid in patternsMetadataTable:
                score = abs(float(patternsMetadataTable[pid][0]))
            else:
                score = 1

            if int(pid) < 0:
                cn.inc(score, currentTime)
                cp.update(currentTime)
            else:
                cp.inc(score, currentTime)
                cn.update(currentTime)

            if pid not in wf:
                cnt = TimeWindowCounter(i);
                cnt.inc(score, currentTime)
                wf[pid] = cnt
            else:
                wf[pid].inc(score, currentTime)


# the program has ended, force a last print
handler(0, None)
