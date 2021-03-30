import os

try:

    os.mkdir('./data', mode=0o755)

except FileExistsError:

    print('data directory already exists')

os.chdir('./data')

try:

    os.mkdir('./helix', mode=0o755)

except FileExistsError:

    print('data/helix directory already exists')

try:

    os.mkdir('./concentric', mode=0o755)

except FileExistsError:

    print('data/concentric directory already exists')

try:

    os.mkdir('./quadratic', mode=0o755)

except FileExistsError:

    print('data/quadratic directory already exists')
