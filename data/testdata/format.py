file = open("test_seinfeld.txt", "r")
output = open("test_seinfeld_stripped.txt", "w")

for x in file:
    a_string = x
    alpha = " "
    for character in x:
        if character.isalnum() or character.isspace():
            alpha += character
    print(alpha)
    output.write(alpha + " ")