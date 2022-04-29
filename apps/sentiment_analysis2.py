import math
import time
from sys import stdin


class expTimeDecayingCounter:
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

    def update(self, timestamp_new):
        decayrate = math.log(2) / self.halflife
        if self.timestamp == 0:
            self.timestamp = timestamp_new

        dt = timestamp_new - self.timestamp

        self.counter = math.exp(-decayrate * dt) * self.counter
        self.timestamp = timestamp_new

    def get(self):
            return self.counter

pat_table = {}
pat_freqs = {}

# both negative and positive words will be stored in the same file
fout = open("patterns.txt", "w")

# load negative words and assign them with a unique negative ID
pat_id=0
with open("../patterns/sentiment/negative_words_en.txt", "r") as fin:
    for line in fin:
        patrn = line.strip('\n')
        pat_id -= 1
        pat_table[pat_id] = patrn
        fout.write(str(pat_id) + " "  "\"" + patrn + "\"" + "\n")
    fin.close()

# load positive words, assign them with a unique positive ID
pat_id=0
with open("../patterns/sentiment/positive_words_en.txt", "r") as fin:
    for line in fin:
        patrn = line.strip('\n')
        pat_id += 1
        pat_table[pat_id] = patrn
        fout.write(str(pat_id) + " "  "\"" + line.strip('\n') + "\"" + "\n")
    fin.close()

# all patterns have been stored
fout.close()

cnt = expTimeDecayingCounter(60) # counter for the last minute
nlines = 0;
nmatches = 0;

def run_command(command):
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


for line in stdin:
    if (line.find("Pattern") == 0):
        nmatches += 1
        cnt.inc(time.time())

        print(nmatches, cnt.get())

        pat_id = line.split()[1]

        if pat_id not in pat_freqs:
            pat_freqs[pat_id] = 0
        else:
            pat_freqs[pat_id] = pat_freqs[pat_id] + 1


print(pat_freqs)
