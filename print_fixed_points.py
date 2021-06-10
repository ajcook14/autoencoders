import gzip, pickle, argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('--n', metavar='n', required=True, type=int, nargs='+',
                    help='layers (including input)')

args = parser.parse_args()
layers = args.n


ls = os.listdir('./data/fixed_points/layers/relu/')

ls_split = list(map(lambda x: x.split('_'), ls))

for i in range(len(ls_split)):

    item = ls_split[i][2]

    item_split = item.split('-')

    if layers == list(map(lambda x: int(x), item_split)):

        fname = ls[i]

        break

print(fname)

f = gzip.open(f'./data/fixed_points/layers/relu/{fname}', 'rb')

fixed_points, layers, seed, limits, activation = pickle.load(f)

f.close()



print(f'max = {max(fixed_points)}')
print(f'lenfp = {len(fixed_points)}')
