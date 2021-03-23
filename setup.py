import os

try:

    os.mkdir('./data', mode=0o755)

except FileExistsError:

    print('data directory already exists')
