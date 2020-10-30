import re
import string
from config import nep_characters 


f = open("stories.txt", "r",encoding="utf8")
txt = f.read()
f.close()

pattern = r"[^{}]".format(nep_characters)

#Replace unnecessary characters
txt = re.sub(r"\n", r" ", txt)
txt = re.sub(pattern,r"",txt)

# Store the cleaned text in the another file
f = open("cleaned.txt", "w",encoding="utf8")
f.write(txt)
f.close()