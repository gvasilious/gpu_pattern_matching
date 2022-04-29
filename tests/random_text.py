import random
import string
import sys

def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


lines_total = 128

if len(sys.argv) > 1:
    lines_total = int(sys.argv[1])

englishWords = open('/usr/share/dict/words').read().splitlines()

words = englishWords

inputFile = open("input.txt", "w")
patrnFile = open("patterns.txt", "w")

flag = False
i = 0
line_length = 0
while i < lines_total or lines_total < 0:

    length = random.randint(1,15)

    word = ""
    if random.randint(0, 5) == 0:
        word = random.choices(words)[0]
    else:
        word = randomString(length)

    if flag == False and len(word) > 3 and random.randint(0, 50) == 10:
        patrnFile.write(word + '\n')
        flag = True

    inputFile.write(word)

    line_length += length

    if (line_length > 60):
        line_length = 0
        i = i + 1
        flag = False
        inputFile.write('\n')
    else:
        inputFile.write(' ')

inputFile.close()
patrnFile.close()
