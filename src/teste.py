
fake = [('white', 40), ('new', 38), ('american', 37), ('’re', 35), ('-', 30), ('old', 27), ('public', 26), ('good', 26), ('right', 24), ('russian', 21), ('dead', 19), ('free', 19), ('republican', 18), ('clear', 18), ('little', 17), ('high', 17), ('able', 16), ('criminal', 16), ('black', 16), ('great', 16)]

real = [('presidential', 101), ('american', 74), ('new', 72), ('political', 71), ('black', 70), ('republican', 66), ('public', 49), ('’re', 49), ('democratic', 48), ('local', 43), ('great', 42), ('-', 42), ('foreign', 41), ('white', 38), ('old', 37), ('right', 37), ('long', 35), ('nuclear', 34), ('federal', 33), ('early', 33)]
# print (fake)
str = ''
for tuple in real:
    str = str + '\hline\n ' + tuple[0] +' & ' + f'{tuple[1]}' + ' \\\\ \n'
# print(str)

# ents_fake = [('Trump', 127), ('Donald Trump', 65), ('Obama', 39), ('White House', 39), ('FBI', 34), ('’s', 31), ('American', 28), ('Republican', 27), ('Americans', 24), ('America', 23), ('U.S.', 23), ('Russia', 22), ('United States', 22), ('Russian', 21), ('Clinton', 21), ('Trump ’s', 19), ('Hillary Clinton', 16), ('today', 15), ('Muslims', 14), ('Friday', 13)]
# ents_real = [('Trump', 380), ('Clinton', 208), ('Donald Trump', 105), ('U.S.', 94), ('Republican', 86), ('Obama', 80), ('New York', 69), ('Hillary Clinton', 67), ('Democrats', 65), ('Monday', 56), ('America', 56), ('’s', 54), ('Senate', 54), ('American', 47), ('Americans', 46), ('Republicans', 46), ('Democratic', 44), ('Buchanan', 44), ('Tuesday', 40), ('CNN', 39)]
ents_fake = [('States', 25), ('United', 24), ('America', 23), ('U.S.', 23), ('Russia', 22), ('New', 21), ('Washington', 14), ('York', 12), ('Florida', 10), ('Minneapolis', 10), ('Obama', 9), ('California', 7), ('Hollywood', 7), ('Australia', 7), ('Idaho', 7), ('Snowden', 7), ('D.C.', 6), ('Texas', 6), ('Chicago', 6), ('North', 6)]
ents_real = [('New', 112), ('U.S.', 94), ('York', 88), ('North', 72), ('Korea', 59), ('America', 58), ('Carolina', 48), ('United', 46), ('States', 44), ('Virginia', 43), ('South', 43), ('Obama', 26), ('Washington', 21), ('West', 21), ('Florida', 20), ('Texas', 20), ('Charlotte', 20), ('City', 18), ('Pennsylvania', 17), ('Jersey', 17)]
                                                                        
str = ''
for tuple in ents_real:
    str = str + '\hline\n ' + tuple[0] +' & ' + f'{tuple[1]}' + ' \\\\ \n'
print(str)
