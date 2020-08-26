nep_characters ="अआइईउऊऋएऎओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहािीुूृेेैोौ्ॉंःँ। "

"""For Testing of the characters purity"""
if __name__ == "__main__":
    print(" " in nep_characters, nep_characters.index(" "), len(nep_characters))
    chars = []
    for c in nep_characters:
        # print(c)
        if c in chars:
            print("repeated", c)
            break    
