import os

try:

    os.mkdir('./data', mode=0o755)

except FileExistsError:

    print('data directory already exists')

try:

    os.mkdir('./figures', mode=0o755)

except FileExistsError:

    print('figures directory already exists')

os.chdir('./figures')

try:

    os.mkdir('./helix', mode=0o755)

except FileExistsError:

    print('figures/helix directory already exists')


try:

    os.mkdir('./concentric', mode=0o755)

except FileExistsError:

    print('figures/concentric directory already exists')

try:

    os.mkdir('./quadratic', mode=0o755)

except FileExistsError:

    print('figures/quadratic directory already exists')

try:

    os.mkdir('./circle', mode=0o755)

except FileExistsError:

    print('figures/circle directory already exists')



os.chdir('../')

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

try:

    os.mkdir('./circle', mode=0o755)

except FileExistsError:

    print('data/circle directory already exists')
