def print_table(data, width='auto', i0=True):
    if not isinstance(width, int):
        if width == 'auto':
            width = []
            for i in data:
                for idx, j in enumerate(i):
                    if idx >= len(width):
                        width += [0]
                    width[idx] = max(len(str(j))+2, width[idx])
        else:
            raise NotImplementedError

    for i, d in enumerate(data):
        line = '|'.join(str(x).ljust(width if isinstance(
            width, int) else width[idx]) for idx, x in enumerate(d))
        print(line)
        if i == 0 and i0:
            print('-' * len(line))
